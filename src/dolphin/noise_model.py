# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import pickle
import logging
import random
import numpy as np
from tqdm import tqdm
from data.transform import logspec

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def get_noise_stats(noise_model_path, noise_snr, dataset, 
                    spectrum_conf, meanstd_norm = None, n = 10000):

    '''Computes or loads mean and std statistics of noise.

    Computed statistics are per-frequency mean and standard deviation 
    of the data. If `noise_model_path` exists, it should contain mean and std 
    as dictionary {'mean': np.array of shape (F,), 'std': np.array of shape F}
    stored in pickle format. Otherwise, the statistics are computed and stored 
    in `noise_model_path`.

    Args:
        noise_model_path (str): Path to load/store statistics.
        noise_snr (tuple): minimum and maximum SNR of noise w.r.t mixture
        dataset (data.JSONAudioDataset): dataset of mixtures which we compute
                                         the SNR with respect to
        spectrum_conf (dict): parameters of STFT
        meanstd_norm (function): normalization to apply to data
        n (int): number of examples of noise to compute the stats from
    '''
    if noise_model_path.exists():
        with open(noise_model_path, 'rb') as f:
            return pickle.load(f)

    logging.info('Did not find noise model, computing it.')
    sums = []
    sums_sq = []
    n_frames = 0

    for i, data in tqdm(enumerate(dataset)):
        if i >= n:
            break

        snr = random.uniform(noise_snr[0], noise_snr[1])
        noise = np.random.randn(*data.shape)
        mix_pow = np.sum(data ** 2)
        noise_pow = np.sum(noise ** 2)
        k = np.sqrt(mix_pow / noise_pow / np.power(10, snr / 10))
        noise *= k

        noise_spec = logspec(noise, **spectrum_conf)
        if meanstd_norm is not None:
            noise_spec = meanstd_norm(noise_spec)

        sums.append(noise_spec.sum(axis = 0))
        sums_sq.append((noise_spec**2).sum(axis = 0))
        n_frames += noise_spec.shape[0]

    stats = {}
    stats['mean'] = np.sum(np.stack(sums), axis = 0) / n_frames
    stats['std'] = np.sum(np.stack(sums_sq), axis = 0) / n_frames
    stats['std'] -= stats['mean']**2
    stats['std'] = np.sqrt(stats['std'])

    stats['mean'] = stats['mean']
    stats['std'] = stats['std']

    noise_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(noise_model_path, 'wb') as f:
        pickle.dump(stats, f)

    return stats


