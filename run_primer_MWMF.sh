#!/bin/bash
#
#SBATCH --job-name=DREAMER
#SBATCH --output=logs/capsnet_%A.out
#SBATCH --error=logs/capsnet_%A.err
#
#SBATCH --nodes=1
#SBATCH --time=50:00:00
#SBATCH --nodelist=komputasi05
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/m450296/miniconda3/envs/virtenv/lib

source ~/miniconda3/etc/profile.d/conda.sh
conda activate virtenv
python capsulenet_primer - Intersubject.py
