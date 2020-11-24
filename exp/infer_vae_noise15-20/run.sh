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


cd /izmolikova/Test/vae_dolphin/exp/infer_vae_noise15-20
export PYTHONPATH=../../src/:$PYTHONPATH

python -m run with i_split=$((SGE_TASK_ID-1)) n_split=600
