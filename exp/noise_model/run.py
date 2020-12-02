# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path
from trainbin.train import noise_models_train

ex = Experiment()

@ex.config
def config():
    dataset = Path('../../data/wsj0-mix_spat/tr_mix.json')
    outdir = Path('models/')
    logdir = Path('./logs')
    ex.observers.append(FileStorageObserver.create(logdir))

    # will be computed if file does not exist
    meanstd_norm_file = Path('../../data/wsj0-mix_spat/tr_meanstd')

    dataset_type = 'JSONAudioDataset'

    nfft = 512
    spectrum_conf = {'sampling_freq': 8000,
                     'window_size': 0.064, # seconds
                     'window_shift': 0.016, # seconds
                     'nfft': nfft}

    snrs = [(15,20), (10,15), (5,10), (0,5), (-5,0)]

@ex.automain
def main(_config):
    noise_models_train(_config)
