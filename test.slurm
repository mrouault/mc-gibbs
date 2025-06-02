#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00
#SBATCH --job-name=test
#SBATCH --mem=1G
#SBATCH --gres=gpu:1 --partition=gpu64
#SBATCH --mail-type=ALL
#SBATCH --mail-user=martin.rouault@univ-lille.fr

python3 examples/ex_thinning.py
