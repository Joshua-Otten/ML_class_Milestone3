#!/bin/bash

#SBATCH --job-name=llamafinetune
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4

#SBATCH --mem=6GB
#SBATCH --time=3-00:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jotten4@gmu.edu

#SBATCH --output=/scratch/jotten4/llama.out.txt
#SBATCH --error=/scratch/jotten4/llama.err.txt

nvidia-smi
python train_model.py

