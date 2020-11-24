# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from dolphin.dolphin import Dolphin, get_noise_model

def sample_normal(mu, logvar):
    return torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu

class VAEDolphin(Dolphin):
    def __init__(self, vae, 
                    n_iterations = 5, 
                    qz_learn_rate = 1e-1, qz_n_updates = 5,
                    n_z_samples = 1,
                    n_speakers = 2, 
                    noise_model = None,
                    device = None, 
                    inline_permutation_aligner = None,
                    wca = -3, 
                    kl_weight = 1):
        '''Dolphin method with VAE as spectral model.

        We use the following notation in documentation:
            C: number of channels 
            S: number of speakers
            N: number of classes (N = S+1)
            T: number of frames
            F: number of frequency bins
        
        Args:
            vae (model.vae.SeqVAESpeaker): trained VAE model
            n_iterations (int): number of iterations of the VB inference
            qz_learn_rate (float): learning rate for updates of params of q(Z)
            qz_n_updates (int): number of updates of params of q(Z) per iteration
            n_z_samples (int): number of samples to approximate expectation w.r.t q(Z)
            n_speakers (int): Number of speakers.
            noise_model (torch.distributions.normal.Normal): trained noise model
            device (torch.device): device to put data on
            inline_permutation_aligner (pb_bss.permutation_alignment.DHTVPermutationAlignment)
            wca (tuple or int): axis with constant weight in spatial model
                                This is effectively used only when initializing
                                spatial model and should not really matter. Kept
                                for consistence with spatial-only inference.
            kl_weight (float): weight \lambda for KL divergence part of loss
        '''
        self.vae = vae
        self.n_iterations = n_iterations
        self.qz_learn_rate = qz_learn_rate
        self.qz_n_updates = qz_n_updates
        self.n_z_samples = n_z_samples
        self.n_speakers = n_speakers
        self.n_classes = n_speakers + 1
        self.noise_model = get_noise_model(noise_model, device)
        self.inline_permutation_aligner = inline_permutation_aligner
        self.device = device
        self.wca = wca
        self.kl_weight = kl_weight

    def initialize_q(self, logspec0):
        '''Initializes q(D) and q(Z) approximate posteriors.

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
        '''
        n_t, n_f = logspec0.shape

        # q(D)
        self.qd = torch.randn((self.n_classes, *logspec0.shape), 
                             dtype = torch.float32)
        self.qd = self.qd / self.qd.sum(dim = 0, keepdims = True)
        self.qd = self.qd.clamp(1e-6, 1 - 1e-6)
        self.qd = self.qd.to(self.device)

        # q(Z): initialized by passing mixture through VAE encoder
        x_in = logspec0.clone()
        latents = self.vae.inference(x_in[None,:,:], [n_t])
        mu_v, logvar_v, mu_u, logvar_u, _, _  = latents
        mu_vs = torch.cat([mu_v for _ in range(self.n_speakers)]).detach()
        logvar_vs = torch.cat([logvar_v for _ in range(self.n_speakers)]).detach()
        mu_us = torch.cat([mu_u for _ in range(self.n_speakers)]).detach()
        logvar_us = torch.cat([logvar_u.clone() for _ in range(self.n_speakers)]).detach()

        mu_vs.requires_grad_()
        mu_us.requires_grad_()
        logvar_vs.requires_grad_()
        logvar_us.requires_grad_()

        self.qz = (mu_vs, logvar_vs, mu_us, logvar_us)

    def update_qZ(self, logspec0):
        '''Updates q(Z) approximate posterior.

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
        '''
        mu_vs, logvar_vs, mu_us, logvar_us = self.qz
        param_to_optimize = (*self.qz,)
        optim_z = optim.SGD(param_to_optimize, lr = self.qz_learn_rate)

        n_t = logspec0.shape[0]

        for i_iter in range(self.qz_n_updates):
            optim_z.zero_grad()

            kls, explogliks = [], []

            for _ in range(self.n_z_samples):
                # sample q(Z1), q(Z2)
                u_samp = sample_normal(mu_us, logvar_us)
                v_samp = sample_normal(mu_vs, logvar_vs)

                # forward Z1, Z2 through VAE decoder
                mu_x, logvar_x = self.vae.emission(v_samp, u_samp, 
                                                [n_t] * self.n_speakers)
                mu_x, logvar_x = mu_x[:,:n_t,:], logvar_x[:,:n_t,:]
                pd_x = Normal(mu_x, torch.exp(0.5 * logvar_x))

                # compute loss Eq.(21)
                kluv, _ = self.vae.kl_divergence_expected_speaker(
                                ((mu_vs, logvar_vs),
                                (mu_us, logvar_us),
                                u_samp, v_samp),
                                torch.tensor([n_t]))
                kls.append(kluv.sum() / self.n_z_samples)
                exploglik = self.qd[:self.n_speakers] * torch.clamp(pd_x.log_prob(logspec0),-14,100)
                exploglik += (1 - self.qd[:self.n_speakers]) * (pd_x.cdf(logspec0) + 1e-6).log()
                explogliks.append(exploglik.sum() / self.n_z_samples)

            pd_x = self.noise_model
            exploglik = self.qd[-1] * torch.clamp(pd_x.log_prob(logspec0),-14,100)
            exploglik += (1 - self.qd[-1]) * (pd_x.cdf(logspec0) + 1e-6).log()
            explogliks.append(exploglik.sum())

            objf = -(torch.sum(torch.stack(explogliks)) - 
                     self.kl_weight * torch.sum(torch.stack(kls)))
            objf.backward()

            for p in param_to_optimize:
                nn.utils.clip_grad_value_(p, 5)
            optim_z.step()

        self.qz = (mu_vs, logvar_vs, mu_us, logvar_us)

    def _spectral_log_p(self, logspec0):
        '''Computes expected conditional log-likelihood in spectral model.

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
        '''
        n_t = logspec0.shape[0]
        mu_vs, logvar_vs, mu_us, logvar_us = self.qz
        pd_xs_all = []
        for i_z in range(self.n_z_samples):
            pd_xs = []
            for i in range(self.n_classes):
                if i < self.n_speakers: # speaker models
                    u_samp1 = sample_normal(mu_us[i], logvar_us[i])
                    v_samp1 = sample_normal(mu_vs[i], logvar_vs[i])

                    mu_x1, logvar_x1 = self.vae.emission(v_samp1[None,...], 
                                                         u_samp1[None,...], 
                                                         [n_t])
                    mu_x1, logvar_x1 = mu_x1[:,:n_t], logvar_x1[:,:n_t]
                    pd_x1 = Normal(mu_x1[0], torch.exp(0.5 * logvar_x1[0]))
                else: # noise model
                    pd_x1 = self.noise_model
                    pd_xs.append(pd_x1)
                pd_xs.append(pd_x1)
            pd_xs_all.append(pd_xs)

        spectral_log_p_all = []
        for i_z in range(self.n_z_samples):
            spectral_log_p = []
            for i in range(self.n_classes):
                qd1_unnorm = pd_xs_all[i_z][i].log_prob(logspec0)
                qd1_unnorm = torch.clamp(qd1_unnorm, -14, 100)
                for j in [jj for jj in range(self.n_classes) if jj != i]:
                    qd1_unnorm += (pd_xs_all[i_z][j].cdf(logspec0) + 1e-6).log()
                spectral_log_p.append(qd1_unnorm)

            spectral_log_p = torch.stack(spectral_log_p) # S, T, F
            spectral_log_p_all.append(spectral_log_p)

        spectral_log_p = torch.stack(spectral_log_p_all).mean(dim = 0)
        return spectral_log_p
