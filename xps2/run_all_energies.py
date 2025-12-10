import os

l_samples = ["gibbs_mala_n2", "gibbs_mala_n3", "gibbs_mh_n2", "gibbs_mh_n3", "kt", "mcmc"]
l_samples_mcmc = ["mcmc"]

for name in l_samples_mcmc:
    s = name
    with open("run_energies_"+s+".slurm", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --ntasks-per-node=10\n")
        file.write("#SBATCH --time=3:00:00\n")
        file.write("#SBATCH --job-name=\"energies_"+s+"\"\n")
        file.write("#SBATCH --output=slurm-energies_"+s+".out\n")
        file.write("#SBATCH --mem=8G\n")

        file.write("ml conda\n")
        file.write("conda activate base\n")
        file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
        file.write("python3 compute_all_energies.py --samples="+s+" \n")
        file.write("deactivate\n")
        file.write("conda deactivate\n")
    os.system("sbatch run_energies_"+s+".slurm")
