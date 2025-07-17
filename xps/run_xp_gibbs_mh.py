import os

#step_sizes_n2_n_10000 = [1e-4, 1e-4, 1e-4, 8e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 5e-7]
#step_sizes_n2_1000_10000 = [5e-4, 2e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 5e-7]
#step_sizes_n2_5000_10000 = [8e-4, 4e-4, 4e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 5e-7]
#step_sizes_n2_1000_50000 = [8e-4, 4e-4, 4e-4, 1e-4, 1e-4, 1e-4, 2e-5, 1e-4, 1e-5, 5e-5]
#step_sizes_n52_1000_10000 = [8e-5, 5e-6, 2e-6, 1e-5, 1e-5, 1e-5, 1e-7, 5e-6, 1e-8, 1e-8]
step_sizes_n3_1000_10000 =[9e-6, 5e-6, 2e-6, 2e-7, 5e-7, 4e-10, 3e-10, 1e-10, 8e-11, 5e-11]

for k in range(1, 11):
    n = 50 * k
    nenv = 1000
    n_iter_gibbs = 10000
    beta_n = n**(3)
    step_size_gibbs = step_sizes_n3_1000_10000[k-1]
    for i in range(100):
        s = str(n)+"_"+str(i)+"_tempn3_env1000_itgibbs10000"
        print(s)
        with open("run_gibbs_mh_"+s+".slurm", "w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --ntasks-per-node=10\n")
            file.write("#SBATCH --time=1:00:00\n")
            file.write("#SBATCH --job-name=\"xp_gibbs_mh_"+s+"\"\n")
            file.write("#SBATCH --output=slurm-gibbs_mh_"+s+".out\n")
            file.write("#SBATCH --mem=32G\n")
        
            file.write("ml conda\n")
            file.write("conda activate base\n")
            file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
            file.write("python3 xp_gibbs_mh.py --key_mcmc=0 --key_gibbs="+str(i)+" --step_size_mcmc_env=1e-1 --step_size_gibbs="+str(step_size_gibbs)+" --n_iter_env="+str(nenv)+" --n_iter_gibbs="+str(n_iter_gibbs)+" --n="+str(n)+" --beta_n="+str(beta_n)+" \n")
            file.write("deactivate\n")
            file.write("conda deactivate\n")
        os.system("sbatch run_gibbs_mh_"+s+".slurm")