# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import soundfile as sf
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

path_to_wsj0_2mix = Path(sys.argv[1]) / 'wsj0-mix'
output_directory = Path(sys.argv[2])

fs = 8000
n_spk = 2
min_max = 'min'
sets = ['tt']
low_snrs = [-5, 0, 5, 10, 15]
high_snrs = [0, 5, 10, 15, 20]

def get_snrs(utt_list, low_snr, high_snr):
    '''
    I was dumb enough not to save the random seeds used in data generation
    for the experiments in the paper. To make the data somewhat consistent,
    the SNRs for each utterance are read from provided lists. In case someone
    tweaks the low_snr, high_snr setting to different values than used 
    in the paper, SNRs are randomly generated.
    '''
    snrs_path = Path('data/snrs') / f'{low_snr}-{high_snr}'
    try:
        with open(snrs_path, 'r') as f:
            snrs = json.load(f)
    except FileNotFoundError:
        logging.warn(f'Did not find {snrs_path}\n'
                     ' -> generating SNRs randomly, '
                     'will not correspond to data used in the paper')
        snrs = {u : np.random.uniform(low_snr, high_snr) 
                for u in utt_list}

    return snrs

def add_noise(data, snr):
    noise = np.random.randn(*data.shape)
    mix_pow = np.sum(data ** 2)
    noise_pow = np.sum(noise ** 2)
    k = np.sqrt(mix_pow / noise_pow / np.power(10, snr / 10))
    noise *= k
    mix_noise = data + noise
    return mix_noise, noise

if not path_to_wsj0_2mix.exists():
    print(f'Path {path_to_wsj0_2mix} does not exist')
    sys.exit(1)

for set in sets:
    path_suff = Path(f'{n_spk}speakers_reverb') / f'wav{int(round(fs / 1000))}k' \
                                                / min_max / set / 'mix'

    if not (path_to_wsj0_2mix / path_suff).exists():
        print(f'path_to_wsj0_2mix {path_to_wsj0_2mix.parent} should include '
              f'sub-directory wsj0-mix/{path_suff}')
        sys.exit(1)

    with open(f'data/wsj0-mix_spat/{set}') as f:
        utt_list = [line.strip() for line in f]

    for low_snr, high_snr in zip(low_snrs, high_snrs):
        outdata_dir = output_directory / f'wsj0-mix_noise_{low_snr}_{high_snr}' \
                      / path_suff
        outdata_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f'Generating {set} {low_snr}-{high_snr} dB')

        snrs = get_snrs(utt_list, low_snr, high_snr)
        with open(output_directory / f'{set}_{low_snr}-{high_snr}_snrs', 'w') as f:
            json.dump(snrs, f, indent=2)

        for utt in tqdm(utt_list):
            for c in range(1,9):
                data, fs_check = sf.read(str(path_to_wsj0_2mix / path_suff / f'{utt}_{c}.wav'))
                assert fs == fs_check, 'Inconsistency in sampling rates'

                snr = snrs[utt]
                mix_noise, noise = add_noise(data, snr)

                sf.write(str(outdata_dir / f'{utt}_{c}.wav'), mix_noise, fs)
