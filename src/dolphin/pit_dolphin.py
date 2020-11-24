# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
from dolphin.dolphin import Dolphin

class PITDolphin(Dolphin):
    def __init__(self, mask, 
                    n_iterations = 5,
                    n_speakers = 2,
                    device = None, 
                    inline_permutation_aligner = None,
                    wca = -3):
        '''Dolphin method with PIT as spectral model.

        We use the following notation in documentation:
            C: number of channels 
            S: number of speakers
            N: number of classes (N = S+1)
            T: number of frames
            F: number of frequency bins
        
        Args:
            mask (np.array): masks returned by PIT network of shape (N,T,F)
            n_iterations (int): number of iterations of the VB inference
            n_speakers (int): Number of speakers.
            device (torch.device): device to put data on
            inline_permutation_aligner (pb_bss.permutation_alignment.DHTVPermutationAlignment)
            wca (tuple or int): axis with constant weight in spatial model
                                This is effectively used only when initializing
                                spatial model and should not really matter. Kept
                                for consistence with spatial-only inference.
                 
        '''
        self.mask = torch.tensor(mask).to(device)
        self.n_iterations = n_iterations
        self.n_speakers = n_speakers
        self.n_classes = n_speakers + 1
        self.inline_permutation_aligner = inline_permutation_aligner
        self.device = device
        self.wca = wca

    def initialize_q(self, logspec0):
        self.qd = torch.tensor(self.mask)

    def update_qZ(self, logspec0):
        pass

    def _spectral_log_p(self, logspec0):
        return (self.mask + 1e-6).log()
