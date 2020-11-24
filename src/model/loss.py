# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
from torch import nn
from torch.nn import functional as F

class Loss(nn.Module):
    def __init__(self):
        '''Base class for implementing different loss functions.'''
        super(Loss, self).__init__()

    def step(self):
        '''Make adjustment of the loss after each epoch.
        
        Could be used for example to anneal weighting of the loss.
        '''
        pass

    def param_to_optimize(self):
        return []

class ELBOSpeakerLoss(Loss):
    def __init__(self, model, speaker_loss_weight):
        '''Combination of classic ELBO loss and speaker classification loss.

        Args:
            model: VAE model which we optimize, 
                   should have `reconstruction_loss`, `kl_divergence`  and 
                   `speaker_loss` methods
            speaker_loss_weight (float)
        '''
        super(ELBOSpeakerLoss, self).__init__()

        self.model = model
        self.speaker_loss_weight = speaker_loss_weight

    def forward(self, inputs, spk_id, lengths, model_outputs):
        '''
        Args:
            inputs: inputs of the model of shape (B,T,F), where
                    B is batch size
                    T is number of frames
                    F is number of frequency bins
            spk_id: tensor of speaker ids of size B
            lengths: tensor of original lengths (before padding) of inputs
            model_outputs: tuple with outputs of the model of form
                           (x, *latents), where x is passed to reconstruction
                           and latents are passed to kl_diveregence
        '''
        x = model_outputs[0]
        latents = model_outputs[1:]
        n_batch, n_frames, x_dim = inputs.shape

        explik = self.model.reconstruction_loss(inputs, x, lengths)
        kl, kl_parts = self.model.kl_divergence(latents, spk_id, lengths)
        elbo_loss = -(explik - kl)
        elbo_parts = {'explik': explik, 'kl': kl, **kl_parts}

        spk, spk_parts = self.model.speaker_loss(spk_id, lengths)

        parts = {'elbo_loss': elbo_loss, 'spkloss': spk, 
                 **elbo_parts, **spk_parts}
        return elbo_loss + self.speaker_loss_weight * spk, parts

    def step(self):
        pass


class ELBOLoss(Loss):
    def __init__(self, model):
        '''Classic ELBO loss for training of VAE.

        Args:
            model: VAE model which we optimize, 
                   should have `reconstruction_loss`, `kl_divergence`
        '''
        super(ELBOLoss, self).__init__()

        self.model = model

    def forward(self, inputs, lengths, model_outputs):
        '''
        Args:
            inputs: inputs of the model of shape (B,T,F), where
                    B is batch size
                    T is number of frames
                    F is number of frequency bins
            lengths: tensor of original lengths (before padding) of inputs
            model_outputs: tuple with outputs of the model of form
                           (x, *latents), where x is passed to reconstruction
                           and latents are passed to kl_diveregence
        '''
        x = model_outputs[0]
        latents = model_outputs[1:]
        n_batch, n_frames, x_dim = inputs.shape

        explik = self.model.reconstruction_loss(inputs, x, lengths)
        kl, kl_parts = self.model.kl_divergence(latents, lengths)

        elbo = explik - kl
        loss = -elbo

        # for monitoring
        parts = {'explik': explik, 'kl': kl, **kl_parts}

        return loss, parts

    def param_to_optimize(self):
        return []

