#!/bin/bash

#SBATCH --job-name=llamatesting
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:2
#SBATCH -N 1
#SBATCH --ntasks-per-node=4

#SBATCH --mem=6GB
#SBATCH --time=0-06:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jotten4@gmu.edu

#SBATCH --output=/scratch/jotten4/testing.out.txt
#SBATCH --error=/scratch/jotten4/testing.err.txt

nvidia-smi
python test_model.py

