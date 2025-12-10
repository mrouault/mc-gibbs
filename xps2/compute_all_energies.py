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

dic_mcmc = pickle.load(open("mcmc/points_mcmc_coverage.p", "rb"))
I_K_pi = dic_mcmc["I_K_pi"]

key_mcmc = 1000 #not used before
step_size = 1e-3
n_iter = 100_000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that specifies parameters"
    )
    parser.add_argument("--samples", required=True, type=str)
    args = parser.parse_args()
    samples_names = args.samples

#-----------------------------------------------------------------------
d = 10

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)
def K_gauss(x, y) :
    return jnp.exp(-0.5*norm_2_safe_for_grad(x-y))

#Target distribution: 10D truncated Gaussian on B(0, 1)
sigma = 0.1
class gaussian_trunc(numpyro.distributions.Distribution) :

    def __init__(self):

        self.d = d
        self.sigma = sigma
        event_shape = (d, )
        super(gaussian_trunc, self).__init__(event_shape = event_shape)

    def outlier(self, value):

        return norm_2_safe_for_grad(value) >= 1.

    def log_prob(self, value) :

        out = self.outlier(value)
        res = cond(out,
                   lambda _ : -jnp.inf,
                   lambda _ : -norm_2_safe_for_grad(value)/(2*self.sigma**2),
                   None)

        return res
    
#------------------------------------------------------
key = random.PRNGKey(key_mcmc)
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= 0.01*jnp.eye(d))
target_mcmc = gaussian_trunc()
sample_mh_jit = jit(vmap(partial(mh,
                            log_prob_target = target_mcmc.log_prob,
                            n_iter = 5_000+n_iter,
                            step_size = step_size)))


#-----------------------------------------
def k_inter_samples(sample_1, sample_2):
    sample_1 = jnp.array(sample_1)
    sample_2 = jnp.array(sample_2)
    n_1 = sample_1.shape[1]
    n_2 = sample_2.shape[1]
    index_1 = jnp.array([k for k in range(n_1)])
    index_2 = jnp.array([k for k in range(n_2)])
    k_inter_func = lambda j ,k : K_gauss(sample_1[:, k], sample_2[:, j])
    pair_j = lambda j : jit(vmap(partial(k_inter_func, j)))(index_2).sum()
    k_inter_sum = jit(vmap(pair_j))(index_1).sum()
    return k_inter_sum
#print(k_inter_samples(sample_1, sample_2))

# assume sample_1: (d, n1), sample_2: (d, n2) 
# K_gauss(x, y) returns scalar

# Blocked sum to avoid out of memory jax errors
def kernel_inter_sum_blocked(sample_1, sample_2, block_size=10_000):
    n1 = sample_1.shape[1]
    n2 = sample_2.shape[1]
    total = 0.0

    X = np.array(sample_1, dtype=np.float64)
    Y = np.array(sample_2, dtype=np.float64)

    for i0 in range(0, n1, block_size):
        i1 = min(n1, i0 + block_size)
        Xb = X[:, i0:i1]

        for j0 in range(0, n2, block_size):
            j1 = min(n2, j0 + block_size)
            Yb = Y[:, j0:j1]

            # pairwise differences
            Kblock= k_inter_samples(Xb, Yb)
            total = total + jnp.sum(Kblock)

    return jnp.exp(jnp.log(total)-jnp.log(n1)-jnp.log(n2))

def compute_energy(sample, key):
    n = sample.shape[1]
    key, _ = random.split(key, 2)
    start_sample_v = mvn.sample(key, (1, ))
    sample_mcmc, log_probs_mcmc, acceptance_mcmc = sample_mh_jit(random.split(key, 1), start_sample_v)
    return kernel_inter_sum_blocked(sample, sample) - 2*kernel_inter_sum_blocked(sample, sample_mcmc[0, 5_000:, :].T) + I_K_pi



#Gibbs
if samples_names == "gibbs_mala_n2":
    paths = ["gibbs_last/last_gibbs_mala_0_0_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_1_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_2_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_3_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mala_0_4_0.001_0.0001_1000_10000_100_10000.0.p"]
    for s in paths:
        print(s)
        dic = pickle.load(open(s, "rb"))
        energies_s = {}
        samples_s = dic["points_gibbs"]
        for k in range(100):
            samples_s_k = samples_s[k]
            energies_s[k] = compute_energy(samples_s_k, key)
        dic["energies"] = energies_s
        pickle.dump(dic, open(s, "wb"))


if samples_names == "gibbs_mala_n3":
    paths = ["gibbs_last/last_gibbs_mala_0_0_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_1_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_2_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_3_0.001_0.0001_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mala_0_4_0.001_0.0001_1000_10000_100_1000000.0.p"]
    for s in paths:
        print(s)
        dic = pickle.load(open(s, "rb"))
        energies_s = {}
        samples_s = dic["points_gibbs"]
        for k in range(100):
            samples_s_k = samples_s[k]
            energies_s[k] = compute_energy(samples_s_k, key)
        dic["energies"] = energies_s
        pickle.dump(dic, open(s, "wb"))

if samples_names == "gibbs_mh_n2":
    paths = ["gibbs_last/last_gibbs_mh_0_0_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_1_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_2_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_3_0.001_0.0001_1000_10000_100_10000.0.p",
            "gibbs_last/last_gibbs_mh_0_4_0.001_0.0001_1000_10000_100_10000.0.p"]
    for s in paths:
        print(s)
        dic = pickle.load(open(s, "rb"))
        energies_s = {}
        samples_s = dic["points_gibbs"]
        for k in range(100):
            samples_s_k = samples_s[k]
            energies_s[k] = compute_energy(samples_s_k, key)
        dic["energies"] = energies_s
        pickle.dump(dic, open(s, "wb"))

if samples_names == "gibbs_mh_n3":
    paths = ["gibbs_last/last_gibbs_mh_0_0_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_1_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_2_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_3_0.001_1e-05_1000_10000_100_1000000.0.p",
            "gibbs_last/last_gibbs_mh_0_4_0.001_1e-05_1000_10000_100_1000000.0.p"]
    for s in paths:
        print(s)
        dic = pickle.load(open(s, "rb"))
        energies_s = {}
        samples_s = dic["points_gibbs"]
        for k in range(100):
            samples_s_k = samples_s[k]
            energies_s[k] = compute_energy(samples_s_k, key)
        dic["energies"] = energies_s
        pickle.dump(dic, open(s, "wb"))

#KT

if samples_names == "kt":
    paths = ["kt/points_thinning_100_0.p",
            "kt/points_thinning_100_1.p",
            "kt/points_thinning_100_2.p",
            "kt/points_thinning_100_3.p",
            "kt/points_thinning_100_4.p"]
    for s in paths:
        print(s)
        dic = pickle.load(open(s, "rb"))
        energies_s = {}
        samples_s = dic["points_thinned"]
        for k in range(100):
            samples_s_k = samples_s[k]
            energies_s[k] = compute_energy(samples_s_k, key)
        dic["energies"] = energies_s
        pickle.dump(dic, open(s, "wb"))

#MCMC

if samples_names == "mcmc":
    s="mcmc/points_mcmc_coverage.p"
    print(s)
    dic = pickle.load(open(s, "rb"))
    energies_s = {100: {}, 1_000:{}, 5_000 :{}, 10_000:{}, 50_000 :{}}
    samples_s = dic["points_mcmc"]
    for N in [100, 1_000, 5_000, 10_000, 50_000]:
    #for N in [50_000]:
        print(N)
        for k in range(100):
            samples_s_k = samples_s[k][:, :N]
            energies_s[N][k] = compute_energy(samples_s_k[:, :], key)
    dic["energies"] = energies_s
    pickle.dump(dic, open(s, "wb"))