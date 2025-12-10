#100 independent MCMC samples and two other independent ones to compute the MMD
#removed 5000 burn in iterations, step size is 1e-3, target is 10 truncated Gaussian on B(0, 1) with variance 0.01


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
import time

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
    parser.add_argument("--step_size", required=True, type=float)
    parser.add_argument("--n_iter", required=True, type=int)
    args = parser.parse_args()
    key_mcmc = args.key_mcmc
    step_size_mh = args.step_size
    n_iter = args.n_iter


#rewrite for the good target


#------------------------------------------------------
#Define kernel, external confinment and target distribution
d = 10

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)


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
step_size_mh = args.step_size #1e-3
dic_mcmc = {"step_size_mh": step_size_mh,
            "n_iter": n_iter,
            "key_mcmc": {},
            "burn_in_mcmc" : 5000,
            "target": "10D truncated Gaussian",
            "sigma": 0.1,
            "d": d,
            "acceptance_mcmc" : {},
            "points_mcmc" : {}}

times = np.ones(102)
for k in range(102): #lauching two more independent chain to compute the energies
    #long mcmc
    t0 = time.time()
    key, _ = random.split(key, 2)
    dic_mcmc["key_mcmc"][k] = key
    start_sample_v = mvn.sample(key, (1, ))
    sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = 5_000+n_iter,
                                step_size = step_size_mh))
    sample_mcmc, log_probs_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key, 1), start_sample_v) #has shape (1, n_iter, d)
    print(acceptance_mcmc)
    dic_mcmc["acceptance_mcmc"][k] = acceptance_mcmc
    dic_mcmc["points_mcmc"][k] = sample_mcmc[0, 5_000:, :].T
    times[k] = time.time()- t0
    print(times[k])
dic_mcmc["Average time"] = np.mean(times)
pickle.dump(dic_mcmc, open("mcmc/points_mcmc_coverage.p", "wb"))
