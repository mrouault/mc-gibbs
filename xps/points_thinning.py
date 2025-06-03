#Imports
#%matplotlib inline
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('..\\.venv\\lib\\site-packages')
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
#mpl.rcParams['text.usetex'] = True
#------------------------------------------------------
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
sigma = 0.1

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
    
#------------------------------------------------------
key = random.PRNGKey(0)
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= jnp.eye(d))
target_mcmc = gaussian_trunc()
step_size_mh = 5e-3

#kernel thinning: select 100 points from 10_000
def kernel_eval(x: Array, y: Array) -> Array:
    eps = 0.1
    s = d-2
    dist_sq =  np.sum((x-y)**2,axis=1) 
    return((eps**2+dist_sq)**(-s/2))
#------------------------------------------------------
l_sizes = [50*k for k in range(1, 11)]
l_m = np.log2(l_sizes).astype('int') #number of halving rounds
l_n_iter_mcmc = [l_sizes[k]* 2**l_m[k] for k in range(len(l_sizes))] #number of mcmc samples
print(l_n_iter_mcmc)
points_thinned = {}
res_thinning = {"step_size_mh": step_size_mh,
                   "l_sizes": l_sizes,
                   "l_m": l_m,
                   "l_n_iter_mcmc": l_n_iter_mcmc,
                   "target": "truncated 3D gaussian",
                   "sigma": sigma,
                   "kernel": "regularized Coulomb",
                   "d": d,
                   "eps": 0.1,
                   "points_thinned" : points_thinned}

for i in range(len(l_sizes)):
    #long mcmc
    key, _ = random.split(key, 2)
    start_sample_v = mvn.sample(key, (1, ))
    sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = l_n_iter_mcmc[i],
                                step_size = step_size_mh))
    sample_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key, 1), start_sample_v) #has shape (1, n_iter_mh, d)
    print(f"Acceptance rate:{acceptance_mcmc}")
    X = sample_mcmc[0, :, :]
    X_thinned = np.zeros((100, d, l_sizes[i]))
    print(X_thinned.shape)
    #kernel thinning
    for j in range(100):
        coreset = kt.thin(X, l_m[i], split_kernel  = kernel_eval, swap_kernel = kernel_eval, delta = 0.5, store_K = True, verbose = True)
        X_thinned[j, :, :] =  X[coreset, :].T
    points_thinned[l_sizes[i]] = X_thinned
    res_thinning['points_thinned'] = points_thinned
    pickle.dump(res_thinning, open('xps/points_thinning.p', 'wb'))