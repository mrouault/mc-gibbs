#######################
#Checking the decay of the Gibbs log prob along the history of the MALA iterates.
#Comparing as well the target log prob of the last MALA iterate and the history of the MCMC chain targeting pi.

# Target is a 10D truncated Gaussian on the unit ball with sigma = 0.1
# Interaction is a Gaussian kernel
# External confinment is approximated with 1000 MCMC samples with 5000 burn in iterations, initialized with a Gaussian distribution with small variance




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

#plot config
import seaborn as sns
sns.set_theme(style="ticks")

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=14)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
plt.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['text.usetex'] = True

dic_gibbs_n2 = pickle.load(open("gibbs_step_size_tuning_mh_n2.p", "rb"))
dic_gibbs_n3 = pickle.load(open("gibbs_step_size_tuning_mh_n3.p", "rb"))

# Gibbs measure log probability"
l = np.array([k for k in range(100)])
l_indexes = np.array([1_000*i for i in range(5, 101)])
gibbs_log_probs_n2 = dic_gibbs_n2["gibbs_log_probs"][0, l_indexes]
gibbs_log_probs_n3 = dic_gibbs_n3["gibbs_log_probs"][0, l_indexes]
#print(gibbs_log_probs_n2[:50])

fig, axes = plt.subplots()
axes.plot(l_indexes, gibbs_log_probs_n2, ls = "--", label = "$\\beta_n = n^2$")
axes.legend()
axes.set_xlabel("Number of iterations")
#axes.set_ylabel("$\\log I_{K}(\\mu_n - \\pi)$")
#axes.set_xscale("log")
#axes.set_yscale("log")
plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
fig.savefig("decay_gibbs_log_prob_n2.pdf")

fig, axes = plt.subplots()
axes.plot(l_indexes, gibbs_log_probs_n3, ls = "--", label = "$\\beta_n = n^3$")
axes.legend()
axes.set_xlabel("Number of iterations")
#axes.set_ylabel("$\\log I_{K}(\\mu_n - \\pi)$")
#axes.set_xscale("log")
#axes.set_yscale("log")
plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
fig.savefig("decay_gibbs_log_prob_n3.pdf")


# MCMC targeting pi log probability
#------------------------------------------------------
#Define target distribution
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


#-------------------------------------------------------
#Initial samples
key_mcmc = random.PRNGKey(0)
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix= 0.1*jnp.eye(d))
target_mcmc = gaussian_trunc()

#------------------------------------------------------
#mcmc sample
n_iter_env = 100_000
step_size_mcmc_env = 1e-3
start_sample_v = mvn.sample(key_mcmc, (1, ))
sample_mh_jit = vmap(partial(mh,
                                log_prob_target = target_mcmc.log_prob,
                                n_iter = n_iter_env,
                                step_size = step_size_mcmc_env))
sample_mcmc, log_probs_mcmc, acceptance_mcmc = jit(sample_mh_jit)(random.split(key_mcmc, 1), start_sample_v) #has shape (1, n_iter_mh, d)
print(acceptance_mcmc)
fig, axes = plt.subplots()
axes.plot(l_indexes, log_probs_mcmc[0, l_indexes], ls = "--", label = "MCMC targeting $\\pi$")
axes.legend()
axes.set_xlabel("Number of iterations")
#axes.set_ylabel("$\\log I_{K}(\\mu_n - \\pi)$")
#axes.set_xscale("log")
#axes.set_yscale("log")
plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
fig.savefig("mcmc_log_prob.pdf")

#plotting log prob of target for the first particle of the MALA iterates targeting the Gibbs measure
samples_gibbs_n2 = dic_gibbs_n3["points_gibbs"] #has shape (n_iter, d, n)
samples_x1 = samples_gibbs_n2[:, :, 1] #first particle along the MALA iterates
l_indexes = np.array([1_000*i for i in range(5, 201)])
log_probs_x1 = jnp.array([target_mcmc.log_prob(samples_x1[i, :]) for i in l_indexes])
print(log_probs_x1)
fig, axes = plt.subplots()
axes.plot(l_indexes, log_probs_x1, ls = "--", label = "First particle along MALA iterates targeting Gibbs")
axes.legend()
axes.set_xlabel("Number of iterations")
#axes.set_ylabel("$\\log I_{K}(\\mu_n - \\pi)$")
#axes.set_xscale("log")
#axes.set_yscale("log")
plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
fig.savefig("gibbs_target_log_prob_first_particle.pdf")