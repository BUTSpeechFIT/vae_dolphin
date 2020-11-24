# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
from torch import nn
from torch import autograd
from torch import optim
import numpy as np
from collections import defaultdict
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class Trainer:
    def __init__(self, model, lossfunc, dataloader, 
                       outdir, device,
                       grad_norm = None,
                       learn_rate = 1e-3,
                       n_epochs = 1000,
                       init_model = None,
                       mode = 'single-speaker-spkid'
                       ):
        '''Wraps the training process of VAE.

        Args:
            model: VAE to train (see model/vae.py)
            lossfunc: loss function (see model/loss.py)
            dataloader: PyTorch dataloader with one of datasets from data/dataset.py
            outdir: directory to store trained model (all epochs are stored)
            device: device to put the data on
            grad_norm (float): value to clip gradient norm at
            learn_rate (float)
            n_epochs (int)
            init_model (str): Path to trained model for initialization or None
                              for random initialization
            mode (str): either `single-speaker` or `single-speaker-spkid` 
                        depending on whether we use dataset and loss function
                        which includes speaker ids
        '''
        self.model = model
        if init_model is not None:
            self.model.load_state_dict(torch.load(init_model, 
                                                  map_location = device))

        self.lossfunc = lossfunc
        self.dataloader = dataloader
        self.outdir = outdir
        self.grad_norm = grad_norm
        self.n_epochs = n_epochs
        self.mode = mode
        self.optimizer = optim.Adam(model.parameters(), lr = learn_rate)
        self.device = device
        self.writer = SummaryWriter() # for tensorboard

    def run(self):
        for epoch in range(1, self.n_epochs + 1):
            logging.info(f'Starting epoch {epoch}')
            self.restart_loss()
            self.run_epoch(epoch)
            torch.save(self.model.state_dict(), self.outdir / f'model.{epoch}')
            self.report_loss(epoch)

    def keep_track_of_loss(self, losses):
        for key in losses:
            self.losses[key].append(losses[key].mean())

    def report_loss(self, epoch):
        '''Reports losses to logging output and Tensorboard.'''
        logging.info(f'Current losses for epoch {epoch}:')
        for key in self.losses:
            loss_mean = np.mean([l.item() for l in self.losses[key]])
            logging.info(f'\t {key}: {loss_mean}') # to output
            self.writer.add_scalar(f'{key}', loss_mean, epoch) # for tensorboard

    def restart_loss(self):
        self.losses = defaultdict(list)

    def run_epoch(self, epoch):
        self.model.train()

        for data in tqdm(self.dataloader):
            if self.mode in ['single-speaker']:
                input, lengths = data
                input = input.to(self.device)
                lengths = lengths.to(self.device)
                loss_inputs = [input, lengths]
            elif self.mode in ['single-speaker-spkid']:
                input, spk_id, lengths = data
                input = input.to(self.device)
                lengths = lengths.to(self.device)
                spk_id = spk_id.to(self.device)
                loss_inputs = [input, spk_id, lengths]
            else:
                raise KeyError(f'Unknown mode {self.mode}')

            self.optimizer.zero_grad()
            model_outputs = self.model(input, lengths)
            loss, loss_parts = self.lossfunc(*loss_inputs, model_outputs)
            loss.mean().backward()

            # for monitoring
            loss_and_parts = {'loss': loss, **loss_parts}
            self.keep_track_of_loss(loss_and_parts)

            if self.grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.grad_norm)

            self.optimizer.step()

        self.lossfunc.step()
