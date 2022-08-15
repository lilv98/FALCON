#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J HPO_test
#SBATCH -o HPO_test.%J.out
#SBATCH -e HPO_test.%J.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[v100]

for lr in 1e-3 1e-4
do
    for emb_dim in 32 64 128
    do
        for num_ng in 4 8 16
        do
            python hpo.py --model FALCON --lr $lr --emb_dim $emb_dim --num_ng $num_ng --verbose 0 --gpu 0
        done
    done 
done
