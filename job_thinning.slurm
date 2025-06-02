#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=11:00:00
#SBATCH --job-name="points_thinning"
#SBATCH --mem=32G
#SBATCH --gres=gpu:1 --partition=gpu64
#SBATCH --mail-type=ALL
#SBATCH --mail-user=martin.rouault@univ-lille.fr

ml conda
conda activate base
source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate
python3 xps/points_thinning.py
deactivate
conda deactivate
