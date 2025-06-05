#Imports
#%matplotlib inline
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('..\\.venv\\lib\\site-packages')
import os
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

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
#mpl.rcParams['text.usetex'] = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that specifies parameters for thinning"
    )
    parser.add_argument("--key_mcmc", required=True, type=int)
    parser.add_argument("--key_thinning", required=True, type=int)
    parser.add_argument("--step_size", required=True, type=float)
    parser.add_argument("--n", required=True, type=int)
    args = parser.parse_args()
    key_mcmc = args.key_mcmc
    key_thinning = args.key_thinning
    step_size_mh = args.step_size
    n = args.n
#------------------------------------------------------
#Define kernel, external confinment and target distribution
d = 3

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)

#kernel for thinning
def kernel_eval(x: Array, y: Array) -> Array:
    eps = 0.1
    s = d-2
    dist_sq =  np.sum((x-y)**2,axis=1)
    return((eps**2+dist_sq)**(-s/2))

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
    
#------------------------------------------------------
key = random.PRNGKey(key_mcmc)
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= jnp.eye(d))
target_mcmc = gaussian_trunc()
step_size_mh = args.step_size #5e-3
m = np.log2(n).astype('int')
n_iter_mh = 2**(np.log2(n).astype('int')) * n

#------------------------------------------------------
#long mcmc
key, _ = random.split(key, 2)
start_sample_v = mvn.sample(key, (1, ))
sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = 5_000+n_iter_mh,
                                step_size = step_size_mh))
sample_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key, 1), start_sample_v) #has shape (1, n_iter_mh, d)
fig, axes = plt.subplots()
print(acceptance_mcmc)

X = sample_mcmc[0, 5_000:, :]
coreset = kt.thin(X, m, split_kernel  = kernel_eval, swap_kernel = kernel_eval, delta = 0.5, store_K = True, verbose = True, seed = key_thinning)
X_thinned = X[coreset, :]
print(X_thinned.shape)
#------------------------------------------------------
#Save results
dic_thinning = {"step_size_mh": step_size_mh,
                "n": n,
                "key_mcmc": key_mcmc,
                "key_thinning": key_thinning,
                "m": m,
                "n_iter_mh": n_iter_mh,
                "burn_in_mcmc" : 5000,
                "target": "truncated 3D gaussian",
                "sigma": sigma,
                "kernel": "regularized Coulomb",
                "d": d,
                "eps": 0.1,
                "acceptance_mcmc" : acceptance_mcmc,
                "points_mcmc" : X.T,
                "points_thinned" : X_thinned.T}
pickle.dump(dic_thinning, open("points_thinning_"+str(n)+"_"+str(key_thinning)+".p", "wb"))
