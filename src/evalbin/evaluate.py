# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import logging
import soundfile as sf
import numpy as np
import json
from tqdm import tqdm
from pb_bss.evaluation.module_mir_eval import _bss_eval_sources_and_noise

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def compute_sdr(config):
    with open(config['dataset'],'r') as f:
        dataset = json.load(f)

    sdr_sum = 0
    sdr_n = 0

    config['resultdir'].mkdir(parents = True, exist_ok = True)
    f_out = open(config['resultdir'] / 'per_utt', 'w')

    for u in tqdm(dataset):
        ref1 = sf.read(dataset[u]['path'][0])[0]
        ref2 = sf.read(dataset[u]['path'][1])[0]

        n = ref1.shape[0]

        try:
            enh1 = sf.read(str(config['outdir'] / f'{u}.0.wav'))[0][:n]
            enh2 = sf.read(str(config['outdir'] / f'{u}.1.wav'))[0][:n]
            enh3 = sf.read(str(config['outdir'] / f'{u}.2.wav'))[0][:n]
        except:
            logging.warn(f'Did not find enhanced files for {u}')
            continue

        sdr,_,_,_ = _bss_eval_sources_and_noise(np.stack((ref1,ref2)),
                                                np.stack((enh1,enh2,enh3)))
        f_out.write(f'{u} {sdr[0]} {sdr[1]}\n')
        sdr_sum += sdr[0]
        sdr_sum += sdr[1]
        sdr_n += 2

    f_out.close()

    logging.info(f'SDR  {sdr_sum / sdr_n}')
    with open(config['resultdir'] / 'avg', 'w') as f:
        f.write(f'{sdr_sum / sdr_n}\n')
