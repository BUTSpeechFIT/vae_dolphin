# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class EmissionConv(nn.Module):
    def __init__(self, hid_dim = 256, v_dim = 20, 
                       u_dim = 20, x_dim = 201,
                       logvar_clamp = [-5,5]):
        '''Emission network aka decoder of the VAE.'''
        super(EmissionConv, self).__init__()

        self.conv1 = nn.ConvTranspose1d(u_dim + v_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.conv2 = nn.ConvTranspose1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.conv3 = nn.ConvTranspose1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 2,
                               padding = 1, output_padding = 1)
        self.conv4 = nn.ConvTranspose1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.conv5 = nn.ConvTranspose1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.fcs = nn.ModuleList([nn.Linear(hid_dim, hid_dim) 
                                  for i in range(4)])

        self.mu_x = nn.Linear(hid_dim, x_dim)
        self.logvar_x = nn.Linear(hid_dim, x_dim)

        self.clamp_min = logvar_clamp[0]
        self.clamp_max = logvar_clamp[1]

    def forward(self, v, u, lengths):
        h = F.relu(self.conv1(torch.cat((v, u), dim = -1).permute(0,2,1)))
        h = F.relu(self.conv2(h)) + h
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h)) + h
        h = F.relu(self.conv5(h)) + h
        h = h.permute(0,2,1)

        for fc in self.fcs:
            h = F.relu(fc(h)) + h

        mu_x = self.mu_x(h)
        logvar_x = self.logvar_x(h)
        logvar_x = torch.clamp(logvar_x, 
                            min = self.clamp_min, 
                            max = self.clamp_max)
        
        return mu_x, logvar_x

class InferenceConv(nn.Module):
    def __init__(self, hid_dim = 400, v_dim = 20, 
                       u_dim = 20, x_dim = 201,
                       logvar_clamp = [-5, 5]):
        '''Inference network aka encoder of the VAE.'''
        super(InferenceConv, self).__init__()

        self.conv1 = nn.Conv1d(x_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.conv2 = nn.Conv1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.conv3 = nn.Conv1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 2,
                               padding = 1)
        self.conv4 = nn.Conv1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.conv5 = nn.Conv1d(hid_dim, hid_dim, 
                               kernel_size = 3, stride = 1,
                               padding = 1)
        self.fcs = nn.ModuleList([nn.Linear(hid_dim, hid_dim) 
                                  for i in range(4)])

        self.mu_u = nn.Linear(hid_dim, u_dim)
        self.logvar_u = nn.Linear(hid_dim, u_dim)
        self.mu_v = nn.Linear(hid_dim, v_dim)
        self.logvar_v = nn.Linear(hid_dim, v_dim)

        self.clamp_min = logvar_clamp[0]
        self.clamp_max = logvar_clamp[1]

    def forward(self, x, lengths):
        h = F.relu(self.conv1(x.permute(0,2,1)))
        h = F.relu(self.conv2(h)) + h
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h)) + h
        h = F.relu(self.conv5(h)) + h
        h = h.permute(0,2,1)

        for fc in self.fcs:
            h = F.relu(fc(h)) + h

        mu_u, logvar_u = self.mu_u(h), self.logvar_u(h)
        mu_v, logvar_v = self.mu_v(h), self.logvar_v(h)
        logvar_u = torch.clamp(logvar_u, 
                               min = self.clamp_min, 
                               max = self.clamp_max)
        
        logvar_v = torch.clamp(logvar_v,
                               min = self.clamp_min, 
                               max = self.clamp_max)

        std_u = torch.exp(0.5 * logvar_u)
        eps_u = torch.randn_like(std_u)
        u = mu_u + eps_u * std_u
        
        std_v = torch.exp(0.5 * logvar_v)
        eps_v = torch.randn_like(std_v)
        v = mu_v + eps_v * std_v
        
        return mu_v, logvar_v, mu_u, logvar_u, v, u

class SeqVAESpeaker(nn.Module):
    def __init__(self, v_dim = 20, u_dim = 20,
                       x_dim = 201, 
                       uv_logvar_clamp = [-5,5],
                       emissnet_hid_dim = 256,
                       infernet_hid_dim = 256,
                       n_speakers = 81):
        super(SeqVAESpeaker, self).__init__()
                
        self.emission = EmissionConv(emissnet_hid_dim, v_dim, u_dim, x_dim) 
        self.inference = InferenceConv(infernet_hid_dim, v_dim, u_dim, x_dim, 
                                       uv_logvar_clamp)

        self.spk_discr_v = nn.Sequential(nn.Linear(v_dim, n_speakers))

        self.speaker_vectors = nn.Parameter(torch.zeros(n_speakers, v_dim))

    def forward(self, x, lengths):
        mu_v, logvar_v, mu_u, logvar_u, v, u = self.inference(x, lengths)
        mu_x, logvar_x = self.emission(v, u, lengths)

        # transposed convolution with stride can have +1 on output
        if x.shape[1] == mu_x.shape[1] - 1:
            mu_x = mu_x[:,:-1,:]
            logvar_x = logvar_x[:,:-1,:]

        return ((mu_x, logvar_x), (mu_v, logvar_v), (mu_u, logvar_u), u, v)

    def reconstruction_loss(self, inputs, x, lengths):
        mu_x, logvar_x = x
        var_x = torch.exp(logvar_x)

        part2 = (inputs - mu_x)**2 / var_x
        part1 = np.log(2 * np.pi) + logvar_x
        explik = -0.5 * (part1 + part2)

        # zero the loss for padding
        for i_b, length in enumerate(lengths):
            explik[i_b, length:, :] = 0

        explik = torch.sum(explik, dim = [1, 2])
        return explik

    def kl_divergence(self, latents, speaker_id, lengths):
        v, u, u_sample, v_sample = latents
        mu_v, logvar_v = v
        mu_u, logvar_u = u
        var_v = torch.exp(logvar_v)
        var_u = torch.exp(logvar_u)

        kl_v = 0.5 * torch.sum(
                (mu_v - self.speaker_vectors[speaker_id][:,None,:])**2 
                + var_v - logvar_v - 1, dim = -1)
        kl_u = 0.5 * torch.sum(mu_u**2 + var_u - logvar_u - 1, dim = -1)

        # zero the loss for padding
        lengths_sub = (lengths + 1) // 2 # lengths of subsampled sequences
        for i_b, length in enumerate(lengths_sub):
            kl_u[i_b, length:] = 0
            kl_v[i_b, length:] = 0

        kl_v = kl_v.sum(dim = 1)
        kl_u = kl_u.sum(dim = 1)
        kl = kl_v + kl_u
        parts = {'kl_v': kl_v, 'kl_u': kl_u} # for monitoring
        return kl, parts

    def kl_divergence_expected_speaker(self, latents, lengths):
        '''KL divergence for unknown spaeker.

        Used during inference when we do not have speaker ids (and speakers can
        be different from those during training. The value of speaker-mean is
        then "learned" to minimized the KL divergence. The optimal value is 
        average of mu_v over time.
        '''
        v, u, u_sample, v_sample = latents
        mu_v, logvar_v = v
        mu_u, logvar_u = u
        var_v = torch.exp(logvar_v)
        var_u = torch.exp(logvar_u)

        kl_v = 0.5 * torch.sum(
                (mu_v - mu_v.mean(dim = 1, keepdims = True))**2 
                + var_v - logvar_v - 1, dim = -1)
        kl_u = 0.5 * torch.sum(mu_u**2 + var_u - logvar_u - 1, dim = -1)

        # zero the loss for padding
        lengths_sub = (lengths + 1) // 2
        for i_b, length in enumerate(lengths_sub):
            kl_u[i_b, length:] = 0
            kl_v[i_b, length:] = 0

        kl_v = kl_v.sum(dim = 1)
        kl_u = kl_u.sum(dim = 1)
        kl = kl_v + kl_u
        parts = {'kl_v': kl_v, 'kl_u': kl_u} # for monitoring
        return kl, parts

    def speaker_loss(self, speaker_id, lengths):
        out_v = self.spk_discr_v(self.speaker_vectors[speaker_id])
        loss_spk = nn.CrossEntropyLoss(reduction = 'none')(out_v, speaker_id)

        return loss_spk, {}

