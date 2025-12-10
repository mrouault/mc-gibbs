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
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
#mpl.rcParams['text.usetex'] = True


#---------------------------------
def coverage(energies, eps):
    #compute the prop that is greater than eps
    n_samples = energies.shape[-1]
    #print(energies)
    prop = 0.
    for k in range(n_samples):
        if energies[k] > eps**2:
            prop+=1.
    return prop / n_samples

def clopper_pearson_ci(x):
    #todo
    return True

#energies is MMD^2, check wether the energy is > eps**2
l_epsilon = np.array([0.01*k for k in range(1, 120)])
dic_coverages = {}
methods = ["gibbs_mala_n2", "gibbs_mala_n3", "gibbs_mh_n2", "gibbs_mh_n3", "kt"]
paths = {"gibbs_mala_n2":   ["gibbs_last/last_gibbs_mala_0_0_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_1_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_2_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_3_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_4_0.001_0.0001_1000_10000_100_10000.0.p"],
        "gibbs_mala_n3": ["gibbs_last/last_gibbs_mala_0_0_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_1_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_2_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_3_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_4_0.001_0.0001_1000_10000_100_1000000.0.p"],
            "gibbs_mh_n2": ["gibbs_last/last_gibbs_mh_0_0_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_1_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_2_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_3_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_4_0.001_0.0001_1000_10000_100_10000.0.p"],
            "gibbs_mh_n3": ["gibbs_last/last_gibbs_mh_0_0_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_1_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_2_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_3_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_4_0.001_1e-05_1000_10000_100_1000000.0.p"],
            "kt": ["kt/points_thinning_100_0.p",
            "kt/points_thinning_100_1.p",
            "kt/points_thinning_100_2.p",
            "kt/points_thinning_100_3.p",
            "kt/points_thinning_100_4.p"]}

for name in methods:
    coverage_s = {}
    for k in range(len(l_epsilon)):
        dic = pickle.load(open(paths[name][0], "rb"))
        eps = l_epsilon[k]
        energies_k = dic["energies"]
        energies_k = jnp.array([energies_k[i] for i in range(100)])
        coverage_s[eps] = coverage(energies_k, eps)
    dic_coverages[name] = coverage_s
    print(name)
    print(dic_coverages[name])

#doing mcmc
path_mcmc = "mcmc/points_mcmc_coverage.p"
dic_mcmc = pickle.load(open(path_mcmc, "rb"))
l_keys = [100, 1_000, 10_000, 50_000]
for j in range(len(l_keys)):
    coverage_s = {}
    for k in range(len(l_epsilon)):
        eps = l_epsilon[k]
        energies_k = dic_mcmc["energies"][l_keys[j]]
        energies_k = jnp.array([energies_k[i] for i in range(100)])
        coverage_s[eps] = coverage(energies_k, eps)
    name = "mcmc_"+str(l_keys[j])
    dic_coverages[name] = coverage_s
    print(name)
    print(dic_coverages[name])

print("coverages computed")
pickle.dump(dic_coverages, open("mcmc/dic_coverages.p", "wb"))

fig, axes = plt.subplots()

for name in dic_coverages.keys():
    l_cov = []
    for k in range(len(l_epsilon)):
        eps = l_epsilon[k]
        l_cov.append(dic_coverages[name][eps])
    axes.plot(l_epsilon, l_cov, label=name)
axes.set_xlabel('eps')
axes.set_ylabel('Coverage')
axes.legend()
plt.savefig("mcmc/coverages_all_methods.pdf")


