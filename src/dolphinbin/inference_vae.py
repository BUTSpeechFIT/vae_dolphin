# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch
import pickle
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pb_bss.permutation_alignment import DHTVPermutationAlignment
from data.transform import logspec_from_spec, spec, inverse_spec
from data.transform import get_meanstd_norm
from data.dataset import JSONAudioMultichannelDataset, JSONAudioDataset
from dolphin.noise_model import get_noise_stats
from dolphin.vae_dolphin import VAEDolphin
from dolphin.beamforming import Beamformer
from model.vae import SeqVAESpeaker

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

    # load dataset and mean-std normalization
    trans = lambda x: spec(x, **config['spectrum_conf'])
    dataset = JSONAudioMultichannelDataset(config['dataset'], 
                                           transform = trans,
                                           i_split = config['i_split'],
                                           n_split = config['n_split'])
    meanstd_norm = get_meanstd_norm(config['meanstd_norm_file'], None)

    # load VAE
    model = SeqVAESpeaker(**config['vae_conf']).to(device)
    model.load_state_dict(torch.load(str(config['model']), 
                          map_location = device))
    model.training = False

    # load noise model
    noise_model = get_noise_stats(config['noise_model'], None, None, None, None)

    for utt in tqdm(dataset.utts):
        # load data
        y = dataset[utt]
        logspec0 = logspec_from_spec(y[0])
        logspec0 = meanstd_norm(logspec0)
        logspec0 = torch.tensor(logspec0.astype('float32')).to(device)

        # initialize permutation alignment
        pa = DHTVPermutationAlignment(
                stft_size = (logspec0.shape[-1] - 1) * 2,
                segment_start = 70, segment_width = 100, segment_shift = 20,
                main_iterations = 20, sub_iterations = 2,
                similarity_metric = 'cos')

        # run the inference
        vae_dolphin = VAEDolphin(model,
                                 noise_model = noise_model,
                                 device = device,
                                 inline_permutation_aligner = pa,
                                 **config['inference_conf']) 
        vae_dolphin.run(logspec0, y)

        # do final permutation alignment
        mask_perm = pa(vae_dolphin.qd.detach().cpu().numpy().transpose(0,2,1))
        mask_perm = mask_perm.transpose(0,2,1)
        mask_perm = np.clip(mask_perm, 1e-6, 1 - 1e-6)

        # dump masks and q(Z): mostly for debug
        if config['dump_masks']:
            (config['outdir'] / 'masks').mkdir(exist_ok = True)
            with open(config['outdir'] / 'masks' / utt, 'wb') as f:
                pickle.dump({'mask': mask_perm}, f)
        if config['dump_qz']:
            (config['outdir'] / 'qz').mkdir(exist_ok = True)
            qz = [x.detach().cpu().numpy() for x in vae_dolphin.qz]
            with open(config['outdir'] / 'qz' / utt, 'wb') as f:
                pickle.dump(qz, f)

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
