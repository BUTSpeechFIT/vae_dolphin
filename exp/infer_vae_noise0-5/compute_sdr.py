from sacred import Experiment
from sacred.observers import FileStorageObserver
import soundfile as sf
import numpy as np
import json
from pathlib import Path
from pb_bss.evaluation.module_mir_eval import _bss_eval_sources_and_noise

from evalbin.evaluate import compute_sdr

ex = Experiment()

@ex.config
def config():
    sset = 'tt'
    outdir = Path(f'Out/{sset}/')
    dataset = f'../../data/wsj0-mix_spat/{sset}_clean.json'
    resultdir = Path(f'Out/sdr/{sset}')

@ex.automain
def main(_config):
    compute_sdr(_config)
