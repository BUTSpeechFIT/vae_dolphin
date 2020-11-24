# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path
from dolphinbin.inference_gmm import run_inference
import json

ex = Experiment()

@ex.config
def config():
    i_split = 0
    n_split = 1
    sset = 'tt'

    dataset = Path(f'../../data/wsj0-mix_spat/{sset}_n1520_mix.json')
    outdir = Path(f'Out/{sset}')
    logdir = Path('./logs')
    modeldir = Path('../../exp/gmm/models')
    model = modeldir / 'GMM.h5'
    ex.observers.append(FileStorageObserver.create(logdir / f'{sset}_{i_split}_{n_split}'))

    use_gpu = False

    nfft = 512
    spectrum_conf = {'sampling_freq': 8000,
                     'window_size': 0.064, # seconds
                     'window_shift': 0.016, # seconds
                     'nfft': nfft}

    inference_conf = {
            'n_iterations': 100,
            'n_speakers': 2,
            }

    noise_model = Path(f'../../exp/noise_model/models/snr_15_20_wonorm')
    noise_snr = (15,20)

    beamforming_conf = {'ban': False}

    dump_masks = False
    dump_qz = False

@ex.automain
def main(_config):
    run_inference(_config)
