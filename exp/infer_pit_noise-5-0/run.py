# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path
from dolphinbin.inference_pit import run_inference
import json

ex = Experiment()

@ex.config
def config():
    i_split = 0
    n_split = 1
    sset = 'tt'

    dataset = Path(f'../../data/wsj0-mix_spat/{sset}_n-50_mix.json')
    outdir = Path(f'Out/{sset}')
    logdir = Path('./logs')
    maskdir = Path('/izmolikova/Tools/'
                   'uPIT-for-speech-separation/cache_1520_n-50')
    ex.observers.append(FileStorageObserver.create(f'logs/logs_{sset}_{i_split}_{n_split}'))

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

    beamforming_conf = {'ban': False}

    dump_masks = False

@ex.automain
def main(_config):
    run_inference(_config)
