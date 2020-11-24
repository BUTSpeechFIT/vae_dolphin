# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import numpy as np
import torch
import logging
import pickle
from scipy.signal import stft, istft
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def spec(signal, window_size = 0.025,
          window_shift = 0.01, sampling_freq = 16000,
          nfft = 400):
    _,_,_spec = stft(signal, 
                    nperseg = int(window_size * sampling_freq),
                    noverlap = int((window_size-window_shift) * sampling_freq),
                    nfft = nfft)
    return _spec.T

def magspec_from_spec(spec):
    return np.abs(spec)

def magspec(signal, window_size = 0.025,
            window_shift = 0.01, sampling_freq = 16000,
            nfft = 400):
    _spec = spec(signal, window_size, window_shift, sampling_freq, nfft)
    _magspec = magspec_from_spec(_spec)
    return _magspec

def logspec_from_spec(spec, eps = 1e-6):
    return np.log10(magspec_from_spec(spec) + eps)

def logspec(signal, eps = 1e-6, window_size = 0.025,
            window_shift = 0.01, sampling_freq = 16000,
            nfft = 400):
    _spec = spec(signal, window_size, window_shift, sampling_freq, nfft)
    _logspec = logspec_from_spec(_spec)
    return _logspec

def inverse_spec(mag, phase, sig_len,
                    eps = 1e-6, window_size = 0.025,
                    window_shift = 0.01, sampling_freq = 16000,
                    nfft = 400):
    mag_phase = mag * np.exp(1j * phase)
    _,sig = istft(mag_phase.T, 
            nperseg = int(window_size * sampling_freq),
            noverlap = int((window_size-window_shift) * sampling_freq),
            nfft = nfft)
    sig = sig[:sig_len]
    return sig

def get_meanstd_norm(path, dataloader):
    '''Computes or loads mean and std statistics of data.

    Computed statistics are per-frequency mean and standard deviation 
    of the data. If `path` exists, it should contain mean and std as tuple of
    numpy arrays, stored in pickle format. Otherwise, the statistics 
    are computed and stored in `path`.

    Args:
        path (str): Path to load/store statistics.
        dataloader: PyTorch dataloader, loading tensors in shape 1xTxF 
                    with T being number of frames and F number of freq bins.

    Returns:
        normalize (function): function to normalize data. Can be used as 
                              `transform` argument for datasets.
    '''
    if path is None:
        return lambda x: x
    try:
        with open(path, 'rb') as f:
            mean, std = pickle.load(f)
    except FileNotFoundError:
        logging.info('Did not find mean and std, computing')

        sum = 0.
        sum_sq = 0.
        n = 0
        for data in tqdm(dataloader):
            sum += data[0].sum(dim = [0, 1])
            sum_sq += (data[0]**2).sum(dim = [0, 1])
            n += data[0].shape[0] * data[0].shape[1]

        mean = sum / n
        std = torch.sqrt(sum_sq / n - mean**2)
        mean, std = mean.detach().numpy(), std.detach().numpy()

        with open(path, 'wb') as f:
            pickle.dump((mean, std), f)

    normalize = lambda x: (x - mean) / std
    return normalize
