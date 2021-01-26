# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import logging
import soundfile as sf
import numpy as np
import json
from tqdm import tqdm
import itertools
#from pb_bss.evaluation.module_mir_eval import _bss_eval_sources_and_noise

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

def _bss_eval_sources_and_noise(reference_sources, estimated_sources):
    """
    Taken from https://github.com/fgnt/pb_bss/blob/master/pb_bss/evaluation/module_mir_eval.py
    and modified to consider permutation according to SDR metric.

    Reference should contain K speakers, whereas estimated sources should
    contain K + 1 estimates. This includes also the noise, to make sure, that
    permutation is calculated correctly, even when noise is confused with a
    speaker.
    :param reference_sources: Time domain signal with shape (K, T)
    :param estimated_sources: Time domain signal with shape (K + 1, T)
    :return: SXRs ignoring noise reconstruction performance
        with shape (K,), where the dimension is the total number of
        speakers in the source signal.
        The selection has length K, so it tells you which estimated channels
        to pick out of the K + 1 channels, to obtain the K interesting
        speakers.
    """
    from mir_eval.separation import _bss_decomp_mtifilt
    from mir_eval.separation import _bss_source_crit
    K, T = reference_sources.shape
    assert estimated_sources.shape == (K + 1, T), estimated_sources.shape

    # Compute criteria for all possible pair matches
    sdr = np.empty((K + 1, K))
    sir = np.empty((K + 1, K))
    sar = np.empty((K + 1, K))

    for j_est in range(K + 1):
        for j_true in range(K):
            s_true, e_spat, e_interf, e_artif = _bss_decomp_mtifilt(
                reference_sources, estimated_sources[j_est], j_true, 512
            )
            sdr[j_est, j_true], sir[j_est, j_true], sar[
                j_est, j_true
            ] = _bss_source_crit(s_true, e_spat, e_interf, e_artif)

    # Select the best ordering, while ignoring the noise reconstruction.
    permutations = list(itertools.permutations(list(range(K + 1)), K))

    mean_sdr = np.empty(len(permutations))
    dum = np.arange(K)
    for (i, permutation) in enumerate(permutations):
        mean_sdr[i] = np.mean(sdr[permutation, dum])

    optimal_selection = permutations[np.argmax(mean_sdr)]
    idx = (optimal_selection, dum)

    return sdr[idx], sir[idx], sar[idx], np.asarray(optimal_selection)
