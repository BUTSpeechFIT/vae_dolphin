# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
import pickle
import numpy as np
import soundfile as sf
from tqdm import tqdm
import scipy.io as sio
from pb_bss.permutation_alignment import DHTVPermutationAlignment
from data.transform import spec, inverse_spec
from data.dataset import JSONAudioMultichannelDataset
from dolphin.pit_dolphin import PITDolphin
from dolphin.beamforming import Beamformer

def run_inference(config):
    config['outdir'].mkdir(parents=True, exist_ok=True)

    # get device
    if config['use_gpu']:
        device = torch.device('cuda')
        # moving a tensor to GPU
        # useful at BUT cluster to prevent someone from getting the same GPU
        fake = torch.Tensor([1]).to(device)
    else:
        device = torch.device('cpu')

    # load dataset
    trans = lambda x: spec(x, **config['spectrum_conf'])
    dataset = JSONAudioMultichannelDataset(config['dataset'], 
                                           transform = trans,
                                           i_split = config['i_split'],
                                           n_split = config['n_split'])

    for utt in tqdm(dataset.utts):
        # load data
        y = dataset[utt]

        # initialize permutation alignment
        pa = DHTVPermutationAlignment(
                stft_size = (y.shape[-1] - 1) * 2,
                segment_start = 70, segment_width = 100, segment_shift = 20,
                main_iterations = 20, sub_iterations = 2,
                similarity_metric = 'cos')

        # load masks for the utterance
        try:
            m1 = sio.loadmat(config['maskdir'] / f'{utt}.spk1.mat')['mask']
            m2 = sio.loadmat(config['maskdir'] / f'{utt}.spk2.mat')['mask']
            m3 = sio.loadmat(config['maskdir'] / f'{utt}.spk3.mat')['mask']
        except sio.matlab.miobase.MatReadError:
            raise KeyError(f'Empty mask for utt {utt}') 
        mask = np.stack((m1, m2, m3))

        # run the inference
        pit_dolphin = PITDolphin(mask,
                                 device = device,
                                 inline_permutation_aligner = pa,
                                 **config['inference_conf']) 
        pit_dolphin.run(None, y)

        # do final permutation alignment
        mask_perm = pa(pit_dolphin.qd.detach().cpu().numpy().transpose(0,2,1))
        mask_perm = mask_perm.transpose(0,2,1)
        mask_perm = np.clip(mask_perm, 1e-6, 1 - 1e-6)

        # dump masks: mostly for debug
        if config['dump_masks']:
            (config['outdir'] / 'masks').mkdir(exist_ok = True)
            with open(config['outdir'] / 'masks' / utt, 'wb') as f:
                pickle.dump({'mask': mask_perm}, f)

        # beamforming and saving the audio
        for tgt_speaker in range(len(mask_perm)):
            target_mask = mask_perm[tgt_speaker]
            interf_mask = 1 - target_mask
            enh = Beamformer(**config['beamforming_conf'])(y,
                                                           target_mask, 
                                                           interf_mask)

            length = y[0].size
            s = inverse_spec(np.abs(enh), np.angle(enh), length,
                             **config['spectrum_conf'])
            s = s / np.max(np.abs(s) + 1e-6)
            sf.write(str(config['outdir'] / f'{utt}.{tgt_speaker}.wav'), s,
                     config['spectrum_conf']['sampling_freq'])
