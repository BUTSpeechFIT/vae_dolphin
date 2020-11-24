# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
import logging
import json
from data.dataset import get_dataset
from data.padding import PadCollate
from data.transform import logspec, get_meanstd_norm
from torch.utils.data import DataLoader
from model.vae import SeqVAESpeaker
from model.loss import ELBOLoss, ELBOSpeakerLoss
from model.trainer import Trainer
from dolphin.noise_model import get_noise_stats

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def vae_train(config):
    config['outdir'].mkdir(parents=True, exist_ok=True)

    # get device
    if config['use_gpu']:
        device = torch.device('cuda')
        # moving a tensor to GPU
        # useful at BUT cluster to prevent someone from getting the same GPU
        fake = torch.Tensor([1]).to(device)
    else:
        device = torch.device('cpu')

    dataset_class = get_dataset(config['dataset_type'])

    # compute or load mean and std of dataset
    trans = lambda x: logspec(x, **config['spectrum_conf'])
    dataset = dataset_class(config['dataset'], transform = trans)
    dataloader_meanstd = DataLoader(dataset)
    meanstd_norm = get_meanstd_norm(config['meanstd_norm_file'],
                                    dataloader_meanstd)

    # load the dataset
    trans = lambda x: meanstd_norm(logspec(x, **config['spectrum_conf']))
    dataset = dataset_class(config['dataset'], transform = trans)
    dataloader_train = DataLoader(dataset,
                                  batch_size = config['batch_size'],
                                  collate_fn = PadCollate(),
                                  shuffle = True)
    
    # create the model
    model = SeqVAESpeaker(**config['vae_conf']).to(device)

    # store model config
    with open(config['outdir'] / 'vae_config', 'w') as f:
        json.dump(config['vae_conf'], f, indent=2)

    # load loss function
    if config['vae_objective'] == 'elbo':
        loss = ELBOLoss(model).to(device)
    elif config['vae_objective'] == 'elbo_speakerid':
        loss = ELBOSpeakerLoss(model, config['speaker_loss_weight']).to(device)
    else:
        raise KeyError(f'Unknown objective {config["vae_objective"]}')

    # run training
    trainer = Trainer(model, loss, dataloader_train, 
                      config['outdir'],
                      device = device,
                      **config['optimizer_conf'])
    trainer.run()

def noise_models_train(config):
    config['outdir'].mkdir(parents=True, exist_ok=True)
    dataset_class = get_dataset(config['dataset_type'])

    # compute or load mean and std of dataset
    trans = lambda x: logspec(x, **config['spectrum_conf'])
    dataset = dataset_class(config['dataset'], transform = trans)
    dataloader_meanstd = DataLoader(dataset)
    meanstd_norm = get_meanstd_norm(config['meanstd_norm_file'],
                                    dataloader_meanstd)

    # load the dataset
    dataset = dataset_class(config['dataset'])
    
    for low, high in config['snrs']:
        for use_norm in [True, False]:
            logging.info(f'Noise model for SNR {low}-{high} dB, use norm: {use_norm}') 
            name = f'snr_{low}_{high}'
            name = f'{name}_wonorm' if not use_norm else name
            get_noise_stats(config['outdir'] / name, (low, high), dataset, 
                            config['spectrum_conf'],
                            meanstd_norm = meanstd_norm if use_norm else None)
