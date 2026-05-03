#Compute the MMD for all samples, runing independent long MCMC for each to compute IK(\mu_n, \pi)
#Imports
#%matplotlib inline
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('..\\.venv\\lib\\site-packages')
import os
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.3'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
import argparse
#os.environ['JAX_PLATFORMS'] = 'cpu'
from typing import Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import scipy
from jax import random, Array, jit, vmap, grad
from jax.tree_util import Partial as partial
from jax.lax import fori_loop, cond, dynamic_slice
import numpyro
#import optax
import jax
#from jax.config import config
from mcmc_samplers import mh
from gibbs_points import gibbs
jax.config.update("jax_enable_x64", True)
import pickle
import numpy as np
from goodpoints import kt
import time

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=15)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
plt.rcParams['axes.unicode_minus'] = False


#---------------------------------
def coverage(energies, eps):
    #compute the prop that is greater than eps
    n_samples = energies.shape[-1]
    #print(energies)
    prop = 0.
    for k in range(n_samples):
        if 0.5*energies[k] > eps**2:#energy is 2 I_K = 2 MMD^2
            prop+=1.
    return prop / n_samples + 1e-5 #to avoid log(0)

def clopper_pearson_ci(x):
    #todo
    return True

l_epsilon = np.array([0.01*k for k in range(20, 40)])  #from 0.2 to 0.4
dic_coverages = {}
methods = ["gibbs_mala_n2", "gibbs_mala_n3", "gibbs_mh_n2", "gibbs_mh_n3", "kt"]
paths = {"gibbs_mala_n2":   ["gibbs_last/last_gibbs_mala_0_0_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_1_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_2_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_3_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_4_1000_10000_100_10000.0.p"],
        "gibbs_mala_n3": ["gibbs_last/last_gibbs_mala_0_0_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_1_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_2_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_3_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_4_1000_10000_100_1000000.0.p"],
            "gibbs_mh_n2": ["gibbs_last/last_gibbs_mh_0_0_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_1_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_2_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_3_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_4_1000_10000_100_10000.0.p"],
            "gibbs_mh_n3": ["gibbs_last/last_gibbs_mh_0_0_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_1_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_2_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_3_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_4_1000_10000_100_1000000.0.p"],
            "kt": ["kt/points_thinning_100_0.p",
            "kt/points_thinning_100_1.p",
            "kt/points_thinning_100_2.p",
            "kt/points_thinning_100_3.p",
            "kt/points_thinning_100_4.p"]}
path_mcmc = "mcmc/points_mcmc_coverage.p"


#computing minimal energies because there are negatives
min_energy = 0.
for name in methods:
    dic = pickle.load(open(paths[name][0], "rb"))
    energies_k = dic["energies"]
    energies_k = jnp.array([energies_k[i] for i in range(100)])
    min_energy_k = np.min(energies_k)
    print(name, " min energy:", min_energy_k)
    if min_energy_k < min_energy:
        min_energy = min_energy_k
l_keys = [100, 1_000, 10_000, 50_000]
for j in range(len(l_keys)):
    dic_mcmc = pickle.load(open(path_mcmc, "rb"))
    energies = dic_mcmc["energies"][l_keys[j]]
    energies = jnp.array([energies[i] for i in range(100)])
    min_energy_k = np.min(energies)
    print("mcmc_"+str(l_keys[j]), " min energy:", min_energy_k)
    if min_energy_k < min_energy:
        min_energy = min_energy_k
print("min energy:", min_energy)

for name in methods:
    coverage_s = {}
    for k in range(len(l_epsilon)):
        dic = pickle.load(open(paths[name][0], "rb"))
        eps = l_epsilon[k]
        energies_k = dic["energies"]
        energies_k = jnp.array([energies_k[i] - min_energy for i in range(100)])
        if k == 0:
            print(energies_k)
        coverage_s[eps] = coverage(energies_k, eps)
    dic_coverages[name] = coverage_s
    print(name)

#doing mcmc
dic_mcmc = pickle.load(open(path_mcmc, "rb"))
l_keys = [100, 1_000, 10_000]
for j in range(len(l_keys)):
    coverage_s = {}
    for k in range(len(l_epsilon)):
        eps = l_epsilon[k]
        energies_k = dic_mcmc["energies"][l_keys[j]]
        energies_k = jnp.array([energies_k[i] - min_energy for i in range(100)])
        if k == 0:
            print(energies_k)
        coverage_s[eps] = coverage(energies_k, eps)
    name = "mcmc_"+str(l_keys[j])
    dic_coverages[name] = coverage_s
    print(name)

print("coverages computed")
pickle.dump(dic_coverages, open("mcmc/dic_coverages.p", "wb"))

fig, axes = plt.subplots()

for name in ["gibbs_mala_n2", "gibbs_mala_n3", "gibbs_mh_n2", "gibbs_mh_n3"]:
    l_cov = []
    for k in range(len(l_epsilon)):
        eps = l_epsilon[k]
        l_cov.append(dic_coverages[name][eps])
    axes.plot(l_epsilon, np.log(l_cov), label=name)
axes.set_xlabel('$\epsilon$')
axes.set_ylabel('Log Coverage')
axes.legend()
plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
plt.savefig("mcmc/coverages_gibbs.pdf")

fig, axes = plt.subplots()

for name in ["gibbs_mala_n2", "kt", "mcmc_100", "mcmc_1000", "mcmc_10000"]:
    l_cov = []
    for k in range(len(l_epsilon)):
        eps = l_epsilon[k]
        l_cov.append(dic_coverages[name][eps])
    if name == "mcmc_1000":
        axes.plot(l_epsilon, np.log(l_cov), alpha = 0.6, label="mcmc_100")
    else:
        axes.plot(l_epsilon, np.log(l_cov), label=name)
axes.set_xlabel('$\epsilon$')
axes.set_ylabel('Log Coverage')
axes.legend()
plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
plt.savefig("mcmc/coverages_gibbs_vs_mcmc.pdf")


