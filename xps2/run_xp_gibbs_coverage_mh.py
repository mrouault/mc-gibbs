import os

#run 5 times the 100 draws of Gibbs measures to compute coverages, with different keys to ensure indepence
#using MH, step size is 1e-4 for beta_n = n**2 and 1e-5 for beta_n = n**3

key_mcmc = 0
step_size_mcmc_env = 1e-3
step_size_gibbs = 1e-4
n_iter_env = 1_000
n_iter_gibbs = 10_000
n = 100
beta_n = n**2

for k in range(5):
    key_gibbs = k
    s = "_coverage_gibbs_mh_n2_"+str(k)
    print(s)
    with open("run_xp_gibbs_"+s+".slurm", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --ntasks-per-node=10\n")
        file.write("#SBATCH --time=3:00:00\n")
        file.write("#SBATCH --job-name=\"xp_gibbs_"+s+"\"\n")
        file.write("#SBATCH --output=slurm-gibbs_"+s+".out\n")
        file.write("#SBATCH --mem=16G\n")

        file.write("ml conda\n")
        file.write("conda activate base\n")
        file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
        file.write("python3 samples_coverage_gibbs_mh.py --key_mcmc="+str(key_mcmc)+" --key_gibbs="+str(key_gibbs)+" --step_size_mcmc_env="+str(step_size_mcmc_env)+" --step_size_gibbs="+str(step_size_gibbs)+" --n_iter_env="+str(n_iter_env)+" --n_iter_gibbs="+str(n_iter_gibbs)+" --n="+str(n)+" --beta_n="+str(beta_n)+" \n")
        file.write("deactivate\n")
        file.write("conda deactivate\n")
    os.system("sbatch run_xp_gibbs_"+s+".slurm")

key_mcmc = 0
step_size_mcmc_env = 1e-3
step_size_gibbs = 1e-5
n_iter_env = 1_000
n_iter_gibbs = 10_000
n = 100
beta_n = n**3

for k in range(5):
    key_gibbs = k
    s = "_coverage_gibbs_mh_n3_"+str(k)
    print(s)
    with open("run_xp_gibbs_"+s+".slurm", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --nodes=1\n")
        file.write("#SBATCH --ntasks-per-node=10\n")
        file.write("#SBATCH --time=3:00:00\n")
        file.write("#SBATCH --job-name=\"xp_gibbs_"+s+"\"\n")
        file.write("#SBATCH --output=slurm-gibbs_"+s+".out\n")
        file.write("#SBATCH --mem=16G\n")

        file.write("ml conda\n")
        file.write("conda activate base\n")
        file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
        file.write("python3 samples_coverage_gibbs_mh.py --key_mcmc="+str(key_mcmc)+" --key_gibbs="+str(key_gibbs)+" --step_size_mcmc_env="+str(step_size_mcmc_env)+" --step_size_gibbs="+str(step_size_gibbs)+" --n_iter_env="+str(n_iter_env)+" --n_iter_gibbs="+str(n_iter_gibbs)+" --n="+str(n)+" --beta_n="+str(beta_n)+" \n")
        file.write("deactivate\n")
        file.write("conda deactivate\n")
    os.system("sbatch run_xp_gibbs_"+s+".slurm")