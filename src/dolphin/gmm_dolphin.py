# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
from torch.distributions.normal import Normal
from dolphin.dolphin import Dolphin, get_noise_model

class GMMDolphin(Dolphin):
    def __init__(self, gmm, 
                    n_iterations = 5, 
                    n_speakers = 2, 
                    noise_model = None,
                    device = None, 
                    inline_permutation_aligner = None,
                    wca = -3):
        '''Dolphin method with GMM as spectral model.

        We use the following notation in documentation:
            C: number of channels 
            S: number of speakers
            N: number of classes (N = S+1)
            T: number of frames
            F: number of frequency bins
        
        Args:
            gmm (dict): trained GMM model in form:
                        {'n_components': Nc,
                         'means': torch.tensor of shape (Nc, F),
                         'stds': torch.tensor of shape (Nc, F),
                         'weights: torch.tensor of shape (Nc,)}
            n_iterations (int): number of iterations of the VB inference
            n_speakers (int): Number of speakers.
            noise_model (torch.distributions.normal.Normal): trained noise model
            device (torch.device): device to put data on
            inline_permutation_aligner (pb_bss.permutation_alignment.DHTVPermutationAlignment)
            wca (tuple or int): axis with constant weight in spatial model
                                This is effectively used only when initializing
                                spatial model and should not really matter. Kept
                                for consistence with spatial-only inference.
                 
        '''
        self.gmm = gmm
        self.n_iterations = n_iterations
        self.n_speakers = n_speakers
        self.n_classes = n_speakers + 1
        self.noise_model = get_noise_model(noise_model, device)
        self.inline_permutation_aligner = inline_permutation_aligner
        self.device = device
        self.wca = wca

    def initialize_q(self, logspec0):
        '''Initializes q(D) and q(Z) approximate posteriors.

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
        '''
        n_t, n_f = logspec0.shape

        # q(D)
        self.qd = torch.rand((self.n_classes, *logspec0.shape), 
                             dtype = torch.float32)
        self.qd = self.qd / self.qd.sum(dim = 0, keepdims = True)
        self.qd = self.qd.clamp(1e-6, 1 - 1e-6)
        self.qd = self.qd.to(self.device)

        # q(Z)
        self.update_qZ(logspec0)

    def update_qZ(self, logspec0):
        '''Updates q(Z) approximate posterior.

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
        '''
        n_t, n_f = logspec0.shape
        self.qz = torch.zeros(self.n_speakers,
                              self.gmm['n_components'],
                              n_t)

        for i in range(self.n_speakers):
            for c in range(self.gmm['n_components']):
                pd_x1 = Normal(self.gmm['means'][c], self.gmm['stds'][c])
                self.qz[i,c] = (self.qd[i] * 
                                    torch.clamp(pd_x1.log_prob(logspec0),
                                                -14, 100)).sum(dim = 1)
                self.qz[i,c] += ((1 - self.qd[i]) * 
                                 (pd_x1.cdf(logspec0) + 1e-6).log()).sum(dim= 1)
                self.qz[i,c] += self.gmm['weights'][c]

        self.qz = self.qz - self.qz.max(dim = 1, keepdim = True)[0]
        self.qz = self.qz.exp()
        self.qz = self.qz / (self.qz.sum(axis = 1, keepdim = True) + 1e-6)
        self.qz = self.qz.clamp(1e-6, 1 - 1e-6)

    def _spectral_log_p(self, logspec0):
        '''Computes expected conditional log-likelihood in spectral model.

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
        '''
        def expected_log_pdf(i):
            if i < self.n_speakers:
                qd1 = torch.zeros_like(logspec0)
                for c in range(self.gmm['n_components']):
                    pd_x = Normal(self.gmm['means'][c], self.gmm['stds'][c])
                    qd1 += self.qz[i,c][:,None] * pd_x.log_prob(logspec0)
            else:
                pd_x = self.noise_model
                qd1 = pd_x.log_prob(logspec0)
            return qd1

        def expected_log_cdf(i):
            if i < self.n_speakers:
                qd1 = torch.zeros_like(logspec0)
                for c in range(self.gmm['n_components']):
                    pd_x = Normal(self.gmm['means'][c], self.gmm['stds'][c])
                    qd1 += self.qz[i,c][:,None] * (pd_x.cdf(logspec0) + 1e-6).log()
            else:
                pd_x = self.noise_model
                qd1 = (pd_x.cdf(logspec0) + 1e-6).log()
            return qd1

        spectral_log_p = []
        for i in range(self.n_classes):
            qd1_unnorm = expected_log_pdf(i)
            for j in [jj for jj in range(self.n_classes) if jj != i]:
                qd1_unnorm += expected_log_cdf(j)
            spectral_log_p.append(qd1_unnorm)

        spectral_log_p = torch.stack(spectral_log_p)
        return spectral_log_p
