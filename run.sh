#!/bin/bash
#
#SBATCH --job-name=DREAMER
#SBATCH --output=logs/capsnet_%A.out
#SBATCH --error=logs/capsnet_%A.err
#
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=komputasi04

source ~/New_DE_CNN/bin/activate
python capsulenet-multi-gpu_4class_ver5.py
