import os

#run 5 times the 100 draws of Gibbs measures to compute coverages, with different keys to ensure indepence
#using MH, step size is 1e-4 for beta_n = n**2 and 1e-5 for beta_n = n**3

key_mcmc = 0
step_size = 1e-3
n_iter = 100_000

for k in range(1):
    key_thinning = k
    s = "_coverage_mcmc_"+str(k)
    print(s)
    with open("run_xp_mcmc_"+s+".slurm", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --ntasks-per-node=10\n")
        file.write("#SBATCH --time=3:00:00\n")
        file.write("#SBATCH --job-name=\"xp_mcmc_"+s+"\"\n")
        file.write("#SBATCH --output=slurm-mcmc_"+s+".out\n")
        file.write("#SBATCH --mem=16G\n")

        file.write("ml conda\n")
        file.write("conda activate base\n")
        file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
        file.write("python3 samples_mcmc_coverage.py --key_mcmc="+str(key_mcmc)+" --step_size="+str(step_size)+" --n_iter="+str(n_iter)+" \n")
        file.write("deactivate\n")
        file.write("conda deactivate\n")
    os.system("sbatch run_xp_mcmc_"+s+".slurm")