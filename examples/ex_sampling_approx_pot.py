#Imports
#%matplotlib inline
import os
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
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('..\\.venv\\lib\\site-packages')
from mcmc_samplers import mh
from gibbs_points import gibbs
jax.config.update("jax_enable_x64", True)
import pickle

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
n_iter = 50_000
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= jnp.eye(d))
mvn_n = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d*n), covariance_matrix= jnp.eye(d*n))
start_sample_v = mvn.sample(key_v, (1, ))
start_sample_gibbs = mvn_n.sample(key_gibbs, (1, ))

#Run MCMC or Gibbs measure and compute the approximated potential
#With MCMC
target_mcmc = gaussian_trunc()
n_iter_mcmc = 500
step_size_mh = 1e-2
sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = 5_000+n_iter_mcmc,
                                step_size = step_size_mh))
sample_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key, 1), start_sample_v) #has shape (1, n_iter_mh, d)
print(acceptance_mcmc)
V_mcmc = lambda x : V_approx(x, sample = sample_mcmc[0, 5_000:, :].T, K = K_riesz, log_w = jnp.zeros(n_iter_mcmc)) #uniform weights, discard some burn in

#With Gibbs measure, $\mu_V$ uniform measure on B(0, 5*sigma)
def V_unif(x, d = d, R = 5*sigma) :
    return V_ext(x) + (d-2)/(2*R**d) * norm_2_safe_for_grad(x)

n_iter_pot = 50_000
n_pot = 500
beta_n_pot = n_pot**2
step_size_pot = 1e-4
mvn_n_pot = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d*n_pot), covariance_matrix= jnp.eye(d*n_pot))
start_sample_gibbs_pot = mvn_n_pot.sample(key_v, (1, ))

target_pot = gibbs(d, n_pot, g, beta_n_pot, V = V_unif)
sample_pot = target_pot.sample(key_v, start_sample_gibbs_pot, n_iter = n_iter_pot, step_size = step_size_pot)
sample_pot_reshaped = sample_pot["samples"].reshape((d, n_pot))
print(sample_pot["acceptance"])
print(target_pot.log_prob(start_sample_gibbs_pot))
print(target_pot.log_prob(sample_pot["samples"]))

log_w = jnp.array([gaussian_trunc().log_prob(sample_pot_reshaped[:, i]) for i in range(n_pot)]).reshape((n_pot, 1)) #w = pi / mu_V with \mu_V uniform
V_quenched = lambda x : V_approx(x, sample = sample_pot_reshaped, K = K_riesz, log_w = log_w)

#----------------------------------------------------------
#Define Gibbs measure and sample
#With MCMC
target = gibbs(d, n, K_riesz, beta_n, V = V_mcmc)
step_size = 1e-4
sample_gibbs = target.sample(key_gibbs, start_sample_gibbs, n_iter = n_iter, step_size = step_size)
sample_mala_reshaped_mh = sample_gibbs["samples"].reshape((d, n))
print(sample_gibbs["acceptance"])
print(target.log_prob(start_sample_gibbs))
print(target.log_prob(sample_gibbs["samples"]))
#Plot points
fig, axes = plt.subplots()
axes.scatter(*sample_mcmc[0, 5_000:, :2].T, alpha = 0.5, s = 10, label = "Background")
axes.scatter(*sample_mala_reshaped_mh[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
axes.legend()
fig.savefig("V_mcmc_g.pdf")

#With Coulomb background
target = gibbs(d, n, K_riesz, beta_n, V = V_quenched)
step_size = 1e-5
sample_gibbs = target.sample(key_gibbs, start_sample_gibbs, n_iter = n_iter, step_size = step_size)
sample_mala_reshaped_gibbs = sample_gibbs["samples"].reshape((d, n))
print(sample_gibbs["acceptance"])
print(target.log_prob(start_sample_gibbs))
print(target.log_prob(sample_gibbs["samples"]))

#Plot points
fig, axes = plt.subplots()
axes.scatter(*sample_pot_reshaped[:2, :], alpha = 0.5, s = 10, label = "Background")
axes.scatter(*sample_mala_reshaped_gibbs[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
axes.legend()
fig.savefig("V_quenched_g.pdf")

fig, axes = plt.subplots()
axes.scatter(*start_sample_gibbs.reshape((d, n))[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("start_sample_quenched.pdf")

#with Stein trick


#long mcmc
key, _ = random.split(key, 2)
start_sample_v = mvn.sample(key, (1, ))
sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = 10_000,
                                step_size = step_size_mh))
sample_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key, 1), start_sample_v) #has shape (1, n_iter_mh, d)
fig, axes = plt.subplots()
print(acceptance_mcmc)
axes.scatter(*sample_mcmc[0, 5_000:5_100, :2].T, alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("mcmc_points.pdf")

#----------------------------------------------------------
points_mcmc = sample_mcmc[0, :, :].T
points_gibbs_mh = sample_mala_reshaped_mh
points_gibbs_gibbs = sample_mala_reshaped_gibbs
points = {"mcmc": points_mcmc, "gibbs_mh": points_gibbs_mh, "gibbs_gibbs": points_gibbs_gibbs}
pickle.dump(points, open("points.p", "wb"))
