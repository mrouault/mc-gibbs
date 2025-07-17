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
                                n_iter = 100_000,
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
    index_2 = jnp.array([k for k in range(N_2)])
    k_inter = lambda j, k : K_riesz(x_1[:, j], x_2[:, k])
    pair_j = lambda j : jit(vmap(partial(k_inter, j)))(index_2).sum()
    pair_inter = jnp.log(jit(vmap(pair_j))(index_1).sum()) - jnp.log(N_1) - jnp.log(N_2)
    return jnp.exp(pair_inter)

def I_riesz(x) :

    return averaged_k_inter(x, x) - 2*averaged_k_inter(x, sample_mcmc_1[0, 10_000:, :].T)#+ I_K_gauss(sample_mh_energy_1, sample_mh_energy_2)


#import args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that specifies parameters"
    )
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--file_prefix", required=True, type=str)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--name_in_dic", required=True, type=str)
    args = parser.parse_args()
    dir = args.dir
    file_prefix = args.file_prefix
    n = args.n
    name_in_dic = args.name_in_dic

#import pickles from args n and dir and prefix in file names
samples_s = jnp.zeros(shape = (100, d, n))
s = file_prefix+str(n)+"_"
i = 0
#dir =  "pickles_gibbs_mh_tempn52_nenv1000_itgibbs10000"
for file in os.listdir(dir):
    if file.startswith(s):
        print(file)
        dic_file = pickle.load(open(dir + "/"+file,  "rb"))
        samples_s = samples_s.at[i, :, :].set(dic_file[name_in_dic])
        i+=1

#compute averaged energies for each n with different sampling methods
def compute_averaged_energy(samples, i_pi): #average energy over 100 independent runs
    #samples should have shape (n_samples, d, n)
    I_list = jnp.array([I_riesz(samples[i, :,  :]+i_pi) for i in range(samples.shape[0])])
    I_averaged = jnp.mean(I_list, axis = 0)
    I_var =  jnp.var(I_list, axis = 0)
    I_med = jnp.median(I_list, axis = 0)
    I_quantile_99 = jnp.quantile(I_list, 0.90, axis = 0)
    return I_averaged, I_var, I_med, I_quantile_99


l_energies = pickle.load(open("energies.p", "rb"))
I_pi = l_energies.get('I_pi', None)
l_energies_dir = l_energies.get(dir, None)
if l_energies_dir == None:
    l_energies[dir] = {}
    l_energies_dir = l_energies[dir]
print(l_energies)
mean, var, med, quantile_90 = compute_averaged_energy(samples_s, I_pi)
l_energies_dir[str(n)] = {'mean': mean, 'var': var, 'med': med, 'quantile_90': quantile_90}
print(l_energies_dir[str(n)])
#print(l_energies)
l_energies[dir] = l_energies_dir

#I_pi = averaged_k_inter(sample_mcmc_1[0, :, :].T, sample_mcmc_2[0, :, :].T)
#l_energies['I_pi'] = I_pi

pickle.dump(l_energies, open("energies.p", "wb"))