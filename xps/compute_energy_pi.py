#Imports
#%matplotlib inline
#!pip install numpyro
#from google.colab import drive
#drive.mount('/content/drive')
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('..\\.venv\\lib\\site-packages')
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import argparse
os.environ['JAX_PLATFORMS'] = 'cpu'
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
jax.config.update("jax_enable_x64", True)
import pickle
import numpy as np
import seaborn as sns
#Define kernel, external confinment and target distribution
d = 3

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)

def g(x, y, s = d-2) : #coulomb
    return jnp.power(norm_2_safe_for_grad(x-y), -s/2)
    # return - 0.5*jnp.log(norm_2_safe_for_grad(x-y) + eps**2)

def K_riesz(x, y, eps = 0.1, s = d-2) : #coulomb
    return jnp.power(norm_2_safe_for_grad(x-y) + eps**2, -s/2)
    # return - 0.5*jnp.log(norm_2_safe_for_grad(x-y) + eps**2)
#Target distribution: 3D truncated Gaussian
sigma = 0.5
class gaussian_trunc(numpyro.distributions.Distribution) :

    def __init__(self):

        self.d = d
        self.sigma = sigma
        event_shape = (d, )
        super(gaussian_trunc, self).__init__(event_shape = event_shape)

    def outlier(self, value):

        return norm_2_safe_for_grad(value) >= (5*self.sigma)**2

    def log_prob(self, value) :

        out = self.outlier(value)
        res = cond(out,
                   lambda _ : -jnp.inf,
                   lambda _ : -norm_2_safe_for_grad(value)/(2*self.sigma**2),
                   None)

        return res
#long MCMCs to approximate the interaction energy, should they be indep between each runs?
key_mcmc = random.PRNGKey(0)
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= jnp.eye(d))
target_mcmc = gaussian_trunc()

key_mcmc_1, key_mcmc_2, _ = random.split(key_mcmc, 3)
sample_mh_jit = jit(vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = 5000+100_000,
                                step_size = 1e-1)))

start_sample_v = mvn.sample(key_mcmc_1, (1, ))
sample_mcmc_1, acceptance_mcmc = sample_mh_jit(random.split(key_mcmc_1, 1), start_sample_v) #has shape (1, n_iter_mh, d)
print(acceptance_mcmc)

start_sample_v = mvn.sample(key_mcmc_2, (1, ))
sample_mcmc_2, acceptance_mcmc = sample_mh_jit(random.split(key_mcmc_2, 1), start_sample_v) #has shape (1, n_iter_mh, d)
print(acceptance_mcmc)

#compute the constant I(pi)
def averaged_k_inter(x_1, x_2):
    #samples should have shape (d, n)
    d, N_1 = x_1.shape
    d, N_2 = x_2.shape
    index_1 = jnp.array([k for k in range(N_1)])
    #N_2 = 100_000
    index_2 = jnp.array([k for k in range(N_2)])
    index_i = jnp.array([[k for k in range(1000*i, 1000*(i+1))] for i in range(0, 90)])
    k_inter = lambda j, k : K_riesz(x_1[:, j], x_2[:, k])
    pair_j_i = lambda k, i : jit(vmap(partial(k_inter, k)))(index_i[i, :]).sum()
    sum_pair = 0.
    for i in range(100):
        print(i)
        pair_i = jit(vmap(partial(pair_j_i, i)))(index_1)
        sum_pair += pair_i.sum()
    pair_inter = jnp.log(sum_pair) - jnp.log(N_1) - jnp.log(N_2)
    return jnp.exp(pair_inter)

l_energies = {}
print(l_energies)


I_pi = averaged_k_inter(sample_mcmc_1[0, 10_000:, :].T, sample_mcmc_2[0, 10_000:, :].T)
l_energies['I_pi'] = I_pi
print(l_energies)

pickle.dump(l_energies, open("energies.p", "wb"))