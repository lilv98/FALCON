#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --reservation=A100
#SBATCH -J FALCON
#SBATCH -o FALCON.%J.out
#SBATCH -e FALCON.%J.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6


python el.py --verbose 0 --valid_interval 1000 --emb_dim 64
