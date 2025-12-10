#Imports
#%matplotlib inline
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
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
sys.path.append('..\\.venv\\lib\\site-packages')
from gibbs_points import gibbs
jax.config.update("jax_enable_x64", True)
import seaborn as sns
import matplotlib as mpl
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
#------------------------------------------------------
# Define kernel and confinment
def norm_2_safe_for_grad(x) :
    """A version of the squared norm that jax can differentiate without error
    """
    return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)

def V(x) :
    """Quadratic external potential
    """
    return 0.5*norm_2_safe_for_grad(x)

def g(x, y, eps = 0) :
    """Regularized Coulomb potential
    """
    return - 0.5*jnp.log(norm_2_safe_for_grad(x-y) + eps**2)
    #return jnp.power(norm_2_safe_for_grad(x-y) + eps**2, -1/2)


#------------------------------------------------------
# Initial sample
key = random.PRNGKey(0)
key, _ = random.split(key, 2)
d = 2
n = 500
beta_n = n**2
n_iter = 5_000
step_size = 1e-6
mvn = numpyro.distributions.MultivariateNormal(loc=jnp.zeros(d*n), covariance_matrix= jnp.eye(d*n))
start_sample = mvn.sample(key, (1, ))

#Define Gibbs measure and sample
target = gibbs(d, n, g, beta_n, V = V)
sample_gibbs = target.sample(key, start_sample, n_iter = n_iter, step_size = step_size)
sample_mala_reshaped = sample_gibbs["samples"].reshape((d, n))
acceptance = sample_gibbs["acceptance"]
print(acceptance)

#Plot points
fig, axes = plt.subplots()
axes.scatter(*sample_mala_reshaped[:2, :], alpha = 0.8, s = 10)
axes.set_aspect('equal',adjustable='box')
axes.set_xlim(-2.5, 2.5)
axes.set_ylim(-2.5, 2.5)
fig.savefig("points_coulomb_n2_unif.pdf")

#------------------------------------------------------




