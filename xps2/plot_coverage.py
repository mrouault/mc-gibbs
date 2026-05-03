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


#---------------------------------
def coverage(energies, eps):
    #compute the prop that is greater than eps
    n_samples = energies.shape[-1]
    prop = 0.
    for k in range(n_samples):
        if energies[k] > eps:
            prop+=1.
    return prop / n_samples

def clopper_pearson_ci(x):
    #todo
    return True


l_epsilon = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
dic_coverages = {}
methods = ["gibbs_mala_n2", "gibbs_mala_n3", "gibbs_mh_n2", "gibbs_mh_n3", "kt", "mcmc_100", "mcmc_1_000", "mcmc_10_000", "mcmc_50_000"]
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

for name in method:
    coverages_s = {}
    for k in range(len(l_epsilon)):
        dic = pickle.load(open(paths[name][k], "rb"))
        eps = l_epsilon[k]
        energies_k = dic["energies"]
        coverage_s[eps] = coverage(energies_k, eps)
    dic_coverages[name] = coverage_s
    print(name)
    print(dic_coverages[name])

#doing mcmc
path_mcmc = "mcmc/points_mcmc_coverage.p"
dic_mcmc = pickle.load(open(path_mcmc, "rb"))
l_keys = ["100", "1_000", "10_000", "50_000"]
for k in range(len(l_keys)):
    coverages_s = {}
    for k in range(len(l_epsilon)):
        eps = l_epsilon[k]
        energies_k = dic_mcmc["energies"][l_keys[k]]
        coverage_s[eps] = coverage(energies_k, eps)
    name = "mcmc_"+l_keys[k]
    dic_coverages[name] = coverage_s
    print(name)
    print(dic_coverages[name])

pickle.dump(dic_coverages, open("mcmc/dic_coverages.p", "wb"))


