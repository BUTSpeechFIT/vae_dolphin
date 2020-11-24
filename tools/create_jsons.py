# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import sys
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

path_to_wsj02mix = Path(sys.argv[1])
path_to_wsj02mix_spat = Path(sys.argv[2])
path_to_wsj02mix_spat_with_noise = Path(sys.argv[3])
outdir = Path('data/wsj0-mix_spat')

outdir.mkdir(parents=True, exist_ok=True)

low_snrs = [-5, 0, 5, 10, 15]
high_snrs = [0, 5, 10, 15, 20]

# single-speaker utterances for training of VAE
logging.info("Generating training json")
subdir = 'wsj0-mix/2speakers_reverb/wav8k/min/tr'
with open('data/wsj0-mix_spat/tr') as f:
    utt_keys = [line.strip() for line in f]

utts = {}
for spk in ['s1', 's2']:
    for utt in utt_keys:
        paths = [path_to_wsj02mix_spat / subdir / spk / f'{utt}_{c}.wav' 
                 for c in range(1,9)]
        utts[f'{utt}_{spk}'] = {
            'speaker': utt[:3] if spk == 's1' else utt.split('_')[2][:3],
            'path': [str(path.absolute()) for path in paths]
            }

with open(outdir / 'tr_1spk.json', 'w') as f:
    json.dump(utts, f, indent=2)

# mixed utterances for training noise model
logging.info("Generating training mixtures json")
subdir = 'wsj0-mix/2speakers_reverb/wav8k/min/tr'
with open('data/wsj0-mix_spat/tr') as f:
    utt_keys = [line.strip() for line in f]

utts = {}
for utt in utt_keys:
    paths = [path_to_wsj02mix_spat / subdir / 'mix' / f'{utt}_{c}.wav' 
                for c in range(1,9)]
    utts[f'{utt}'] = {
        'path': [str(path.absolute()) for path in paths]
        }

with open(outdir / 'tr_mix.json', 'w') as f:
    json.dump(utts, f, indent=2)

# mixed utterances with noise for testing
with open('data/wsj0-mix_spat/tt') as f:
    utt_keys = [line.strip() for line in f]

for low, high in zip(low_snrs, high_snrs):
    logging.info(f"Generating testing json for noise {low}-{high} dB.")
    subdir = f'wsj0-mix_noise_{low}_{high}/2speakers_reverb/wav8k/min/tt/mix'
    utts = {}
    for utt in utt_keys:
        paths = [path_to_wsj02mix_spat_with_noise / subdir / f'{utt}_{c}.wav' 
                 for c in range(1,9)]
        utts[utt] = {
            'path': [str(path.absolute()) for path in paths]
            }

    with open(outdir / f'tt_n{low}{high}_mix.json', 'w') as f:
        json.dump(utts, f, indent=2)

# clean utterances for evaluation
logging.info("Generating evaluation json")
subdir = 'data/2speakers/wav8k/min/tt'
with open('data/wsj0-mix_spat/tt') as f:
    utt_keys = [line.strip() for line in f]

utts = {}
for utt in utt_keys:
    path1 = path_to_wsj02mix / subdir / 's1' / f'{utt}.wav'
    path2 = path_to_wsj02mix / subdir / 's2' / f'{utt}.wav'
    utts[utt] = {
            'path': [str(path1.absolute()),
                     str(path2.absolute())]
            }
with open(outdir / 'tt_clean.json', 'w') as f:
    json.dump(utts, f, indent=2)
