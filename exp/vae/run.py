# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path
from trainbin.train import vae_train


ex = Experiment()

@ex.config
def config():
    dataset = Path('../../data/wsj0-mix_spat/tr_1spk.json')
    outdir = Path('models/')
    logdir = Path('./logs')
    ex.observers.append(FileStorageObserver.create(logdir))

    # will be computed if file does not exist
    meanstd_norm_file = Path('../../data/wsj0-mix_spat/tr_meanstd')

    dataset_type = 'JSONAudioSpeakerDataset'
    batch_size = 32

    vae_objective = 'elbo_speakerid'
    speaker_loss_weight = 1

    nfft = 512
    spectrum_conf = {'sampling_freq': 8000,
                     'window_size': 0.064, # seconds
                     'window_shift': 0.016, # seconds
                     'nfft': nfft}

    vae_conf = {
        'v_dim' : 20,
        'u_dim' : 20,
        'emissnet_hid_dim' : 512,
        'infernet_hid_dim' : 512,
        'n_speakers': 101,
        'x_dim': nfft // 2 + 1
        }

    optimizer_conf = {
        'learn_rate' : 1e-4,
        'grad_norm' : 10,
        'n_epochs' : 100,
        'mode': 'single-speaker-spkid'
    }

    use_gpu = True

@ex.automain
def main(_config):
    vae_train(_config)
