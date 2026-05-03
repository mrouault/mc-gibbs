#Compute I_{K}(\pi), target is 10 truncated Gaussian on B(0, 2\sigma)


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

def norm_2_safe_for_grad(x) :
      return jnp.sum(x**2)
def K_gauss(x, y) :
    return jnp.exp(-0.5*norm_2_safe_for_grad(x-y))

dic_mcmc = pickle.load(open("mcmc/points_mcmc_coverage.p", "rb"))
key_mcmc = 1_000
step_size = 0.1
n_iter = 1_000_000

#-----------------------------------------------------------------------
d = 10

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)
def K_gauss(x, y) :
    return jnp.exp(-0.5*norm_2_safe_for_grad(x-y))

#Target distribution: 10D truncated Gaussian on B(0, 1)
sigma = 1.
class gaussian_trunc(numpyro.distributions.Distribution) :

    def __init__(self):

        self.d = d
        self.sigma = sigma
        event_shape = (d, )
        super(gaussian_trunc, self).__init__(event_shape = event_shape)

    def outlier(self, value):

        return norm_2_safe_for_grad(value) >= (2*self.sigma)**2

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
start_sample_v = mvn.sample(key, (1, ))
sample_mcmc_1, log_probs_mcmc, acceptance_mcmc = sample_mh_jit(random.split(key, 1), start_sample_v)
key, _ = random.split(key, 2)
sample_mcmc_2, log_probs_mcmc, acceptance_mcmc = sample_mh_jit(random.split(key, 1), start_sample_v)
print(acceptance_mcmc)

key, _ = random.split(key, 2)
#sub-sampling mcmc points to compute IK(\mu_n, \pi)
indices_1 = random.choice(key, 1_000_000, shape=(100_000,))
key, _ = random.split(key, 2)
indices_2 = random.choice(key, 1_000_000, shape=(100_000,))
sample_1 = sample_mcmc_1[0, 5_000+indices_1, :].T
sample_2 = sample_mcmc_2[0, 5_000+indices_2, :].T
print(sample_1.shape)

n_1 = sample_1.shape[1]
n_2 = sample_2.shape[1]

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

# Blocked sum
def kernel_inter_sum_blocked(sample_1, sample_2, block_size=10000):
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

    return total


k_inter = kernel_inter_sum_blocked(sample_1, sample_2, block_size=10_000)
print(k_inter)

I_K = np.exp(np.log(k_inter) - np.log(n_1)-np.log(n_2))
print("Interaction energy: "+str(I_K))

dic_mcmc["I_K_pi"] = I_K
pickle.dump(dic_mcmc, open("mcmc/points_mcmc_coverage.p", "wb"))
