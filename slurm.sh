#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=train
#SBATCH --account=def-lilimou
#SBATCH --nodes=1
#SBATCH --mem=48000M
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100l:1

module load gcc/9.3.0 cuda/11.8.0 cudacore/.11.8.0 cudnn/8.6.0.163 arrow/10.0.1 python/3.10 opencv

pip install -r requirements.txt 

python rainbow.py Alien-v5 Boxing-v5 Breakout-v5 --optimizer=cadam --beta0 0.6 