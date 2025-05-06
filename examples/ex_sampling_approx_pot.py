#Imports
#%matplotlib inline
from typing import Callable
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
import sys
sys.path.append('..')
sys.path.append('.')
from mcmc_samplers import mh
from gibbs_points import gibbs
jax.config.update("jax_enable_x64", True)
#------------------------------------------------------
#Define kernel, external confinment and target distribution
d = 3

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)

def g(x, y, eps = 0., s = d-2) : #coulomb
    return jnp.power(norm_2_safe_for_grad(x-y) + eps**2, -s/2)
    # return - 0.5*jnp.log(norm_2_safe_for_grad(x-y) + eps**2)

#Target distribution: 2D truncated Gaussian
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
key = random.PRNGKey(0)
key_gibbs, key_v, _  = random.split(key, 3)
n = 100
beta_n = n**2
n_iter = 10_000
step_size = 1e-3* (beta_n)**(-1)
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= jnp.eye(d))
mvn_n = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d*n), covariance_matrix= jnp.eye(d*n))
start_sample_v = mvn.sample(key_v, (1, ))
start_sample_gibbs_pot = mvn_n.sample(key_v, (1, ))
start_sample_gibbs = mvn_n.sample(key_gibbs, (1, ))

#Run MCMC or Gibbs measure and compute the approximated potential
#With MCMC
target_mcmc = gaussian_trunc()
n_iter_mcmc = n
step_size_mh = 1e-1
sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = n_iter_mcmc,
                                step_size = step_size_mh))
sample_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key, 1), start_sample_v) #has shape (1, n_iter_mh, d)
print(acceptance_mcmc)
V_mcmc = lambda x : V_approx(x, sample = sample_mcmc[0, :, :].T, K = g, log_w = jnp.zeros(n_iter_mcmc)) #uniform weights

#With Gibbs measure, $\mu_V$ uniform measure on B(0, 5*sigma)
def V_unif(x, d = d, R = 5*sigma) :
    return (d-2)/(2*R**d) * norm_2_safe_for_grad(x)

n_iter_pot = 10_000
step_size_pot = 0.1* (beta_n)**(-1)
target_pot = gibbs(d, n, g, beta_n, V = V_unif)
sample_pot = target_pot.sample(key_v, start_sample_gibbs_pot, n_iter = n_iter_pot, step_size = step_size_pot)
sample_pot_reshaped = sample_pot["samples"].reshape((d, n))
print(sample_pot["acceptance"])

log_w = jnp.array([gaussian_trunc().log_prob(sample_pot_reshaped[:, i]) for i in range(n)]).reshape((n, 1)) #w = pi / mu_V
V_quenched = lambda x : V_approx(x, sample = sample_pot_reshaped, K = g, log_w = log_w)

#Plot points for pot approximation
fig, axes = plt.subplots() 
axes.scatter(*sample_pot_reshaped[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("V_quenched_pot.pdf")

fig, axes = plt.subplots() 
axes.scatter(*sample_mcmc[0, :, :2].T, alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("V_mcmc_pot.pdf")


#----------------------------------------------------------
#Define Gibbs measure and sample
#With MCMC
target = gibbs(d, n, g, beta_n, V = V_mcmc)
sample_gibbs = target.sample(key_gibbs, start_sample_gibbs, n_iter = n_iter, step_size = step_size)
sample_mala_reshaped = sample_gibbs["samples"].reshape((d, n))
print(sample_gibbs["acceptance"])
#Plot points
fig, axes = plt.subplots() 
axes.scatter(*sample_mala_reshaped[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("V_mcmc.pdf")

#With Gibbs
target = gibbs(d, n, g, beta_n, V = V_quenched)
sample_gibbs = target.sample(key_gibbs, start_sample_gibbs, n_iter = n_iter, step_size = step_size)
sample_mala_reshaped = sample_gibbs["samples"].reshape((d, n))
print(sample_gibbs["acceptance"])
#Plot points
fig, axes = plt.subplots() 
axes.scatter(*sample_mala_reshaped[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("V_quenched.pdf")

#print(target.log_prob(start_sample_gibbs))
#print(target.log_prob(sample_gibbs["samples"]))


fig, axes = plt.subplots() 
axes.scatter(*start_sample_gibbs.reshape((d, n))[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("start_sample.pdf")