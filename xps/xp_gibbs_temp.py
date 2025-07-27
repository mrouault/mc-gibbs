#Imports
#%matplotlib inline
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
from gibbs_points import gibbs
jax.config.update("jax_enable_x64", True)
import pickle
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that specifies parameters for thinning"
    )
    parser.add_argument("--key_mcmc", required=True, type=int)
    parser.add_argument("--key_gibbs", required=True, type=int)
    parser.add_argument("--step_size_mcmc_env", required=True, type=float)
    parser.add_argument("--step_size_gibbs", required=True, type=float)
    parser.add_argument("--n_iter_env", required=True, type=int)
    parser.add_argument("--n_iter_gibbs", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--beta_n", required=True, type=float)
    args = parser.parse_args()
    key_mcmc = args.key_mcmc
    key_gibbs = args.key_gibbs
    step_size_mcmc_env = args.step_size_mcmc_env
    step_size_gibbs = args.step_size_gibbs
    n_iter_env = args.n_iter_env
    n_iter_gibbs = args.n_iter_gibbs
    n = args.n
    beta_n = args.beta_n

#------------------------------------------------------
#Define kernel, external confinment and target distribution
d = 3

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)

def g(x, y, s = d-2) : #coulomb
    return jnp.power(norm_2_safe_for_grad(x-y), -s/2)
    # return - 0.5*jnp.log(norm_2_safe_for_grad(x-y) + eps**2)

def K_riesz(x, y, eps = 0.1, s = d-2) : #coulomb
    #return jnp.power(norm_2_safe_for_grad(x-y) + eps**2, -s/2)
    return jnp.exp(-norm_2_safe_for_grad(x-y))
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

#Define approximated V
def V_ext(x) :
    R_2 = (5* sigma)**2
    outlier = norm_2_safe_for_grad(x) >= R_2
    res = jnp.where(outlier, norm_2_safe_for_grad(x) - R_2, 0.)
    return res

def V_approx(x, sample, K, log_w):#using logs in the weights for numerical stability
    d, N_mh = sample.shape
    k_inter = lambda j : log_w[j] + jnp.log(K(x, sample[:, j]))
    v_inter = scipy.special.logsumexp(jit(vmap(k_inter))(jnp.array([k for k in range(N_mh)]))) - scipy.special.logsumexp(log_w)

    return V_ext(x) - jnp.exp(v_inter)

#-------------------------------------------------------
#Initial samples
key_mcmc = random.PRNGKey(key_mcmc)
key_gibbs_mala = random.PRNGKey(key_gibbs)
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= jnp.eye(d))
mvn_n = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d*n), covariance_matrix= jnp.eye(d*n))
target_mcmc = gaussian_trunc()

#------------------------------------------------------
#mcmc sample
start_sample_v = mvn.sample(key_mcmc, (1, ))
sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = 5_000+n_iter_env,
                                step_size = step_size_mcmc_env))
sample_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key_mcmc, 1), start_sample_v) #has shape (1, n_iter_mh, d)
print(acceptance_mcmc)
V_mcmc = lambda x : V_approx(x, sample = sample_mcmc[0, 5_000:, :].T, K = K_riesz, log_w = jnp.zeros(n_iter_env)) #uniform weights, discard some burn in

#------------------------------------------------------
#gibbs sample
start_sample_gibbs = mvn_n.sample(key_gibbs_mala, (1, ))
target = gibbs(d, n, K_riesz, beta_n, V = V_mcmc)
sample_gibbs = target.sample(key_gibbs_mala, start_sample_gibbs, n_iter = n_iter_gibbs, step_size = step_size_gibbs)
sample_mala_reshaped_mh = sample_gibbs["samples"].reshape((d, n))
acceptance_gibbs = sample_gibbs["acceptance"]
print(acceptance_gibbs)
print(target.log_prob(start_sample_gibbs))
print(target.log_prob(sample_gibbs["samples"]))

#------------------------------------------------------
#Save results
dic_gibbs = {
    "step_size_mcmc_env": step_size_mcmc_env,
    "step_size_gibbs": step_size_gibbs,
    "n_iter_env": n_iter_env,
    "n_iter_gibbs": n_iter_gibbs,
    "burn_in_mcmc": 5_000,
    "n": n,
    "beta_n": beta_n,
    "key_mcmc": key_mcmc,
    "key_gibbs": key_gibbs,
    "target": "truncated 3D gaussian",
    "sigma": sigma,
    "kernel": "regularized Coulomb",
    "d": d,
    "eps": 0.1,
    "acceptance_mcmc": acceptance_mcmc,
    "acceptance_gibbs": acceptance_gibbs,
    "points_mcmc": sample_mcmc[0, 5_000:, :].T,  # Discard burn-in,
    "points_gibbs": sample_mala_reshaped_mh
}

s = str(n)+"_"+str(key_gibbs)+"_tempn2_env1000_itgibbs50000"
pickle.dump(dic_gibbs, open("points_gibbs_mh_"+s+".p", "wb"))

fig, axes = plt.subplots()
axes.scatter(*sample_mala_reshaped_mh[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("points_gibbs_gaussian_kernel.pdf")