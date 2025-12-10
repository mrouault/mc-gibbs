import os


with open("run_energy_target.slurm", "w") as file:
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --nodes=1\n")
    file.write("#SBATCH --ntasks-per-node=10\n")
    file.write("#SBATCH --time=30:00\n")
    file.write("#SBATCH --job-name=\"energy_target"+"\"\n")
    file.write("#SBATCH --output=slurm-energy_target"+".out\n")
    file.write("#SBATCH --mem=8G\n")

    file.write("ml conda\n")
    file.write("conda activate base\n")
    file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
    file.write("python3 compute_energy_target.py"+" \n")
    file.write("deactivate\n")
    file.write("conda deactivate\n")
os.system("sbatch run_energy_target.slurm")