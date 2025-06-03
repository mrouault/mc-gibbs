import os

for k in range(1, 8):
    n = 50 * k
    for i in range(100):
        print("n = ", n, " i = ", i)
        with open("run_thinning_"+str(n)+"_"+str(i)+".slurm", "w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --ntasks-per-node=1\n")
            file.write("#SBATCH --time=2:00:00\n")
            file.write("#SBATCH --job-name=\"xp_thinning_"+str(n)+"_"+str(i)+"\"\n")
            file.write("#SBATCH --output=slurm-thinning_"+str(n)+"_"+str(i)+".out\n")
            file.write("#SBATCH --mem=64G\n")
        
            file.write("ml conda\n")
            file.write("conda activate base\n")
            file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
            file.write("python3 xp_thinning.py --key_mcmc=0 --key_thinning="+str(i)+" --step_size=1e-1 --n="+str(n)+" \n")
            file.write("deactivate\n")
            file.write("conda deactivate\n")
        os.system("sbatch run_thinning_"+str(n)+"_"+str(i)+".slurm")

