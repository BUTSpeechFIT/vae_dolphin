# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import numpy as np
from pb_bss.extraction import beamformer as bf

def mvdr(y, target_mask, interf_mask, ban = True):
    '''Runs MVDR beamformer and optional blind analytic postfilter.

    MVDR beamformer is derived from target and interfering masks, 
    which are used to compute spatial correlation matrices and subsequentally,
    the filter coefficients. See pb_bss library for more information.

    In following shapes
        C is number of channels
        T is number of frames
        F is number of frequency bins

    Args:
        y: observation of shape (C, T, F),
        target_mask: mask corresponding to T-F bins dominated 
                     by target speaker, of shape (T, F)
        interf_mask: mask corresponding to T-F bins dominated 
                     by the interfering speakers or noise, of shape (T, F)
        ban: if True, blind analytic postfilter is applied after beamforming

    Returns:
        enh: output of beamforming of shape (T,F)
    '''

    # get COV matrices
    cov_x = bf.get_power_spectral_density_matrix(y.transpose((2,0,1)), 
                                                 target_mask.T)
    cov_n = bf.get_power_spectral_density_matrix(y.transpose((2,0,1)), 
                                                interf_mask.T)

    # get mvdr vector souden
    w_mvdr = bf.get_mvdr_vector_souden(cov_x, cov_n, eps = 1e-6)
    if ban:
        w_mvdr = bf.blind_analytic_normalization(w_mvdr, cov_n)

    # do beamforming
    enh = bf.apply_beamforming_vector(w_mvdr, y.transpose((2,0,1))).T

    return enh
    

class Beamformer:
    '''Wrapper for beamformer calls.

    Now this is very thin wrapper for MVDR call, optionally
    with T-F masking. In future, it could be extended with options
    for more type of beamformers.
    '''
    def __init__(self, ban = True):
        self.ban = ban

    def __call__(self, y, target_mask, interf_mask):
        '''Applies beamforming.

        In following shapes
            C is number of channels
            T is number of frames
            F is number of frequency bins

        Arguments:
            y: observation of shape (C, T, F)
            target_mask: mask corresponding to T-F bins dominated 
                        by target speaker, of shape (T, F)
            interf_mask: mask corresponding to T-F bins dominated 
                        by the interfering speakers or noise, of shape (T, F)
            ban: if True, blind analytic postfilter is applied after beamforming

        '''
        enh = mvdr(y, target_mask, interf_mask, ban = self.ban)

        return enh
