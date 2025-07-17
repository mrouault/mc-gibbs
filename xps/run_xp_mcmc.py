import os

l = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
l2 = [100, 500, 1000, 10000]
for k in l2:
    n = k
    for i in range(500):
        print("n = ", n, " i = ", i)
        with open("run_mcmc_"+str(n)+"_"+str(i)+".slurm", "w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --ntasks-per-node=1\n")
            file.write("#SBATCH --time=10:00\n")
            file.write("#SBATCH --job-name=\"xp_mcmc_"+str(n)+"_"+str(i)+"\"\n")
            file.write("#SBATCH --output=slurm-mcmc_"+str(n)+"_"+str(i)+".out\n")
            file.write("#SBATCH --mem=16G\n")
        
            file.write("ml conda\n")
            file.write("conda activate base\n")
            file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
            file.write("python3 xp_mcmc.py --key_mcmc="+str(i)+" --step_size=1e-1 --n="+str(n)+" \n")
            file.write("deactivate\n")
            file.write("conda deactivate\n")
        os.system("sbatch run_mcmc_"+str(n)+"_"+str(i)+".slurm")