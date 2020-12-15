# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
from torch.distributions.normal import Normal
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
from munkres import Munkres, make_cost_matrix
from pb_bss.distribution import CACGMMTrainer
from pb_bss.distribution.complex_angular_central_gaussian import normalize_observation

def get_noise_model(noise_model, device):
    mean = torch.tensor(noise_model['mean'], dtype = torch.float32).to(device)
    std = torch.tensor(noise_model['std'], dtype = torch.float32).to(device)
    return Normal(mean, std)

class Dolphin:
    '''Base class for different implementations of Dolphin method.

    Implements functionality independent of chosen spectral model:
    * spatial model initialization and update
    * permuations needed during the inference
    * update of q(D) - this however depends on implementation of spectral model
    * overall inference algorithm

    We use the following notation in documentation:
        C: number of channels 
        S: number of speakers
        N: number of classes (N = S+1)
        T: number of frames
        F: number of frequency bins
    '''

    def initialize_spatial(self, spatial_feats):
        '''Initializes spatial model by fitting cacGMM to spatial features.

        Args:
            spatial_feats (np.array): Spatial features of shape (C,T,F)
        '''
        self.cacGMM = CACGMMTrainer()
        spatial_model = self.cacGMM.fit(spatial_feats.transpose(2,1,0),
                                num_classes = self.n_classes,
                                iterations = 1,
                                weight_constant_axis = self.wca,
                                inline_permutation_aligner = self.inline_permutation_aligner,
                                )
        self.spatial = spatial_model

    def permute_global(self, logspec0, spatial_feats): 
        '''Flips the order of classes in q(D) according to spectral and spatial likelihoods.

        The goal is to align components (coresponding to speakers and noise) 
        between spectral and spatial model. We choose to keep the order
        in spectral model and change the order in spatial model. Due to the way
        this function is called (after initializing spatial models and 
        before update of q(Z) and q(D)), this is esentially the same as changing 
        the order in q(D).

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
            spatial_feats (np.array): Spatial features of shape (C,T,F)
        '''
        spatial_norm = normalize_observation(spatial_feats.transpose(2,1,0))
        spat_log_p, _ = self.spatial.cacg._log_pdf(spatial_norm[:,None,:,:])
        spat_log_p = spat_log_p.transpose(1,2,0)

        spectral_log_p = self._spectral_log_p(logspec0)

        perm = self._find_best_permutation(spectral_log_p.detach().numpy(), 
                                           spat_log_p, idx_constant = (1,2))
        self.qd = self.qd[perm[0,0]]

    def _find_best_permutation(self, spectral, spatial, idx_constant):
        '''Finds the best permutation of classes in spectral and spatial model.

        Args:
            spectral (np.array): Conditional log-likelihood of spectral model 
                                 (as in Eq.19) in [1], shape (N,T,F)
            spatial (np.array): Conditional log-likelihood of spatial model 
                                (as in Eq.19) in [1], shape (N,T,F)
            idx_constant (tuple or int) indices of axis which have constant permutation
                Examples:
                    idx_constant = (1,2) 
                        -> for all time frames and frequency bins 
                           the permutation is constant 
                           (finding 1 global permutation)
                    idx_constant = 1
                        -> for all time frames the permutation is constant
                           (finding permutation for each frequency)

        Returns:
            permutations (dict): mapping tuples of time and frequency indices
                                 to the best permutation. For constant indices, 
                                 the map contains only index 0.
                Examples:
                    permutations = {(0,0) : [2, 0, 1]}
                        -> one global permutation (idx_constant = (1,2))
                        -> spectral comp. 0 corresponds to spatial comp. 2
                        -> spectral comp. 1 corresponds to spatial comp. 0
                        -> spectral comp. 2 corresponds to spatial comp. 1

        [1] Integration of variational autoencoder and spatial clustering 
            for adaptive multi-channel neural speech separation; 
            K. Zmolikova, M. Delcroix, L. Burget, T. Nakatani, J. Cernocky
        '''
        if isinstance(idx_constant, int):
            idx_constant = (idx_constant,)
        idx_constant = tuple([i+1 for i in idx_constant])

        perm_scores = logsumexp(spectral[:,None,:,:] + spatial[None,:,:,:],
                                axis = idx_constant)
        perm_scores = np.expand_dims(perm_scores, idx_constant)

        permutations = {}
        for i1, i2 in np.ndindex(perm_scores.shape[-2:]):
            idx_perm = Munkres().compute(make_cost_matrix(perm_scores[:,:,i1,i2]))
            idx_perm.sort(key = lambda x: x[0])
            permutations[i1,i2] = [i[1] for i in idx_perm]

        return permutations

    def update_qD(self, logspec0, spatial_feats): 
        '''Updates q(D) approximate posterior.

        Follows Eq. (19) from [1]. Additionally permutes the spatial components
        to best fit the spectral components at each frequency.
        
        [1] Integration of variational autoencoder and spatial clustering 
            for adaptive multi-channel neural speech separation; 
            K. Zmolikova, M. Delcroix, L. Burget, T. Nakatani, J. Cernocky

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
            spatial_feats (np.array): Spatial features of shape (C,T,F)
        '''
        _, n_t, n_f = spatial_feats.shape
        spatial_norm = normalize_observation(spatial_feats.transpose(2,1,0))
        spat_log_p, _ = self.spatial.cacg._log_pdf(spatial_norm[:,None,:,:])
        spat_log_p = spat_log_p.transpose(1,2,0)

        spectral_log_p = self._spectral_log_p(logspec0)

        perm = self._find_best_permutation(spectral_log_p.detach().numpy(), 
                                           spat_log_p,
                                           idx_constant = 1)
        for f in range(n_f):
            spat_log_p[...,f] = spat_log_p[perm[0,f],:,f]

        ln_qds_unnorm = []
        for i in range(self.n_classes):
            qd1_unnorm = torch.tensor(spat_log_p[i]).to(self.device).float()
            qd1_unnorm += spectral_log_p[i]
            ln_qds_unnorm.append(qd1_unnorm)

        ln_qds_unnorm = torch.stack(ln_qds_unnorm)
        # subtract max for stability of exp (the constant does not matter)
        ln_qds_unnorm = (ln_qds_unnorm - 
                         ln_qds_unnorm.max(dim = 0, keepdim = True)[0])
        qds_unnorm = ln_qds_unnorm.exp()

        qd = qds_unnorm / (qds_unnorm.sum(axis = 0) + 1e-6)
        qd = qd.clamp(1e-6, 1 - 1e-6)
        self.qd = qd.detach()

    def update_spatial(self, spatial_feats):
        '''Updates parameters of spatial model.

        Args:
            spatial_feats (np.array): Spatial features of shape (C,T,F)
        '''
        spatial_norm = normalize_observation(spatial_feats.transpose(2,1,0))
        _, quadratic_form = self.spatial.predict(spatial_feats.transpose(2,1,0),
                                                  return_quadratic_form = True)

        spec_norm = spatial_norm
        spatial_model = self.cacGMM._m_step(spec_norm,
                                        quadratic_form,
                                        self.qd.permute(2,0,1).detach().cpu().numpy(),
                                        saliency = None,
                                        hermitize = True,
                                        covariance_norm = 'eigenvalue',
                                        eigenvalue_floor = 1e-10,
                                        weight_constant_axis = self.wca
                                        )
        self.spatial = spatial_model

    def run(self, logspec0, spatial_feats):
        '''Runs the overall inference algorithm.

        Args:
            logspec0 (torch.Tensor): Spectral features of shape (T,F)
            spatial_feats (np.array): Spatial features of shape (C,T,F)
        '''
        self.initialize_q(logspec0)
        self.initialize_spatial(spatial_feats)
        self.permute_global(logspec0, spatial_feats)

        for i in tqdm(range(self.n_iterations)):
            self.update_qZ(logspec0)
            self.update_qD(logspec0, spatial_feats) 
            self.update_spatial(spatial_feats)

    def initialize_q(self, logspec0):
        raise NotImplementedError

    def update_qZ(self, logspec0):
        raise NotImplementedError

    def _spectral_log_p(self, logspec0):
        raise NotImplementedError
