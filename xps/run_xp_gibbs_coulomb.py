import os


#step_sizes_n2_10000_envn_n2_10000 =[(8e-5, 3e-5), (5e-5, 3e-5), (5e-5, 1e-5), (1e-5, 1e-5), (1e-5, 1e-5), (1e-5, 1e-5), (1e-5,1e-5),
#                                    (1e-5, 1e-5), (1e-5, 1e-6), (2e-6, 5e-7)]
step_sizes_n3_10000_envn_n2_10000 =[(8e-5, 1e-7), (5e-5, 5e-9), (5e-5, 5e-9), (1e-5, 2e-9), (1e-5, 9e-10), (1e-5, 1e-10), (1e-5,1e-10),
                                    (1e-5, 1e-10), (1e-5, 1e-10), (2e-6, 6e-11)]
for k in range(1, 11):
    n = 50 * k
    n_env = n
    beta_n_env = n_env**2
    n_iter_env = 10000
    n_iter_gibbs = 10000
    beta_n = n**(3)
    step_size_env_coulomb, step_size_gibbs = step_sizes_n3_10000_envn_n2_10000[k-1]
    for i in range(100):
        s = str(n)+"_"+str(i)+"_tempn3_itgibbs10000_envn__tempenvn3_itgibbsenv10000"
        print(s)
        with open("run_gibbs_coulomb_"+s+".slurm", "w") as file:
            file.write("#!/bin/bash\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --ntasks-per-node=10\n")
            file.write("#SBATCH --time=1:00:00\n")
            file.write("#SBATCH --job-name=\"xp_gibbs_coulomb_"+s+"\"\n")
            file.write("#SBATCH --output=slurm-gibbs_coulomb_"+s+".out\n")
            file.write("#SBATCH --mem=32G\n")
        
            file.write("ml conda\n")
            file.write("conda activate base\n")
            file.write("source /home/martin.rouault/.cache/pypoetry/virtualenvs/mc-gibbs-TZN4J5XP-py3.11/bin/activate\n")
            file.write("python3 xp_gibbs_coulomb.py --key_env=0 --key_gibbs="+str(i)+" --step_size_env_coulomb="+str(step_size_env_coulomb)+" --step_size_gibbs="+str(step_size_gibbs)+" --n_env="+str(n_env)+" --beta_n_env="+str(beta_n_env)+" --n_iter_env="+str(n_iter_env)+" --n_iter_gibbs="+str(n_iter_gibbs)+" --n="+str(n)+" --beta_n="+str(beta_n)+" \n")
            file.write("deactivate\n")
            file.write("conda deactivate\n")
        os.system("sbatch run_gibbs_coulomb_"+s+".slurm")