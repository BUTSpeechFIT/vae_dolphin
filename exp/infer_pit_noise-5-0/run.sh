#!/bin/bash
#
#$ -S /bin/bash
#$ -N dolphin
#$ -o /izmolikova/vae_dolphin/sge_logs/dolphin.out
#$ -e /izmolikova/vae_dolphin/sge_logs/dolphin.err
#$ -q all.q@@blade
#$ -t 1-600
#$ -tc 100
#


cd /izmolikova/vae_dolphin/exp/infer_pit_noise-5-0
export PYTHONPATH=../../src/:$PYTHONPATH

python -m run with i_split=$((SGE_TASK_ID-1)) n_split=600
