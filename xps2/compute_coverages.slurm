#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=1:00:00
#SBATCH --job-name=compute_coverages
#SBATCH --output=compute_coverages.out
#SBATCH --mem=2G

ml conda
conda activate base
source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate
python3 compute_coverages.py
deactivate
conda deactivate