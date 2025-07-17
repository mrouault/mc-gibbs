import os

l2 = [100, 500, 1000, 10000]
for k in l2:
    n = k
    for i in range(500):
        print("n = ", n, " i = ", i)
        os.system("python3 xp_mcmc.py --key_mcmc="+str(i)+" --step_size=1e-1 --n="+str(n)+" \n")