#Sample 100 independent realisations of a Gibbs measure with temperature beta_n and n = 100 and computing average run time
#target is a 10D truncated Gaussian on the unit ball with sigma = 0.1
#Interaction is a Gaussian kernel
#The external confinment is approximated with 1000 MCMC samples from the target distribution with 5000 burn in iterations

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
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that specifies parameters"
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
d = 10

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)

def g(x, y, s = d-2) : #coulomb
    return jnp.power(norm_2_safe_for_grad(x-y), -s/2)
    # return - 0.5*jnp.log(norm_2_safe_for_grad(x-y) + eps**2)

def K_riesz(x, y, eps = 0.1, s = d-2) : #coulomb
    return jnp.power(norm_2_safe_for_grad(x-y) + eps**2, -s/2)
    #return jnp.exp(-norm_2_safe_for_grad(x-y))
    # return - 0.5*jnp.log(norm_2_safe_for_grad(x-y) + eps**2)

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

#Define approximated V
def V_ext(x) :
    #R_2 = (5* sigma)**2
    R_2 = 1.
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

mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= 0.01*jnp.eye(d))
target_mcmc = gaussian_trunc()

#Save results
dic_gibbs = {
    "step_size_mcmc_env": step_size_mcmc_env,
    "step_size_gibbs": step_size_gibbs,
    "n_iter_env": n_iter_env,
    "n_iter_gibbs": n_iter_gibbs,
    "burn_in_mcmc": 5_000,
    "n": n,
    "beta_n": beta_n,
    "key_mcmc": {},
    "key_gibbs": {},
    "target": "truncated 10D gaussian",
    "sigma": sigma,
    "kernel": "Gaussian kernel",
    "d": d,
    "eps": 1.,
    "acceptance_mcmc": {},
    "acceptance_gibbs": {},
    "points_mcmc": {},  # Discard burn-in,
    "points_gibbs": {},
    "gibbs_log_probs": {},
    "mcmc_log_probs": {}
}

#####################################################
#Runing 100 independent MALA chains starting at the input keys
key_mcmc = random.PRNGKey(key_mcmc)
key_gibbs_mala = random.PRNGKey(key_gibbs)
times = np.zeros(100)
for k in range(100) :
    key_mcmc, _ = random.split(key_mcmc, 2)
    key_gibbs_mala, _ = random.split(key_gibbs_mala, 2)
    dic_gibbs["key_mcmc"][k] = key_mcmc
    dic_gibbs["key_gibbs"][k] = key_gibbs_mala

    #timing start
    start_time = time.time()
    #------------------------------------------------------
    #mcmc sample
    start_sample_v = mvn.sample(key_mcmc, (1, ))
    sample_mh_jit = vmap(partial(mh,
                                    log_prob_target = target_mcmc.log_prob,
                                    n_iter = 5_000+n_iter_env,
                                    step_size = step_size_mcmc_env))
    sample_mcmc, log_probs_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key_mcmc, 1), start_sample_v) #has shape (1, n_iter_mh, d)
    print(acceptance_mcmc)
    V_mcmc = lambda x : V_approx(x, sample = sample_mcmc[0, 5_000:, :].T, K = K_gauss, log_w = jnp.zeros(n_iter_env)) #uniform weights, discard some burn in
    dic_gibbs["acceptance_mcmc"][k] = acceptance_mcmc
    dic_gibbs["mcmc_log_probs"][k] = log_probs_mcmc
    dic_gibbs["points_mcmc"][k] = sample_mcmc[0, 5_000:, :].T  # Discard burn-in
    #------------------------------------------------------
    #gibbs sample
    start_sample_gibbs = mvn.sample(key_gibbs_mala, (n, ))
    start_sample_gibbs_reshaped = start_sample_gibbs.T.reshape((1, d*n))
    target = gibbs(d, n, K_gauss, beta_n, V = V_mcmc)
    sample_gibbs = target.sample(key_gibbs_mala, start_sample_gibbs_reshaped, n_iter = n_iter_gibbs, step_size = step_size_gibbs, method = 'mh', stable = False)

    sample_mala_reshaped_mh = sample_gibbs["samples"].reshape((n_iter_gibbs, d, n))
    acceptance_gibbs = sample_gibbs["acceptance"]
    log_probs_gibbs = sample_gibbs["log_probs"]
    print(acceptance_gibbs)
    dic_gibbs["acceptance_gibbs"][k] = acceptance_gibbs
    dic_gibbs["points_gibbs"][k] = sample_mala_reshaped_mh
    dic_gibbs["gibbs_log_probs"][k] = log_probs_gibbs
    
    #timing end
    end_time = time.time()
    times[k] = end_time - start_time
    print("Run ", k, " time: ", times[k])


    #print(target.log_prob_stable(jnp.atleast_2d(start_sample_gibbs)))

#print(target.log_prob_stable(sample_gibbs["samples"][:, -1, :]))
dic_gibbs["averaged_time"] = np.mean(times)
print("Averaged time over runs: ", dic_gibbs["averaged_time"])
#------------------------------------------------------
#Save results

s = str(args.key_mcmc)+"_"+str(args.key_gibbs)+"_"+str(step_size_mcmc_env)+"_"+str(step_size_gibbs)+"_"+str(n_iter_env)+"_"+str(n_iter_gibbs)+"_"+str(n)+"_"+str(beta_n)
pickle.dump(dic_gibbs, open("gibbs_mh_"+s+".p", "wb"))