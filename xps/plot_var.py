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
sns.set_theme(style="ticks")

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=14)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
#mpl.rcParams['text.usetex'] = True

l_var = pickle.load(open('var.p', 'rb'))
l = [50*k for k in range(1, 11)]
l_n = [50*k for k in range(1, 8)]
l_gibbs_mh_tempn2_env1000_itgibbs10000 = np.array([l_var['pickles_gibbs_mh_tempn2_env1000_itgibbs10000'][str(k)] for k in l])
l_gibbs_mh_tempn2_env5000_itgibbs10000 = np.array([l_var['pickles_gibbs_mh_tempn2_env5000_itgibbs10000'][str(k)] for k in l])
l_gibbs_mh_tempn2_envn_itgibbs10000 = np.array([l_var['pickles_gibbs_mh_tempn2_envn_itgibbs10000'][str(k)] for k in l])
l_gibbs_mh_tempn2_nenv1000_itgibbs50000 = np.array([l_var['pickles_gibbs_mh_tempn2_nenv1000_itgibbs50000'][str(k)] for k in l])
l_thinning =  np.array([l_var['pickles_thinning'][str(k)] for k in l_n])
l_gibbs_mh_tempn52_nenv1000_itgibbs10000 = np.array([l_var['pickles_gibbs_mh_tempn52_nenv1000_itgibbs10000'][str(k)] for k in l])
l_gibbs_mh_temp_n3_env1000_itgibbs10000 = np.array([l_var['pickles_gibbs_mh_temp_n3_env1000_itgibbs10000'][str(k)] for k in l])
l_mcmc = np.array([l_var['pickles_mcmc'][str(k)] for k in l])
l_gibbs_coulomb_tempn2 = np.array([l_var['pickles_gibbs_coulomb_tempn2'][str(k)] for k in l])
l_gibbs_coulomb_tempn3 = np.array([l_var['pickles_gibbs_coulomb_tempn3'][str(k)] for k in l])

fig, axes = plt.subplots()
axes.loglog(l, l_mcmc, ls = "--", marker = "o", markersize = 6, label = "MCMC")
#axes.loglog(l, l_gibbs_mh_tempn2_nenv1000_itgibbs50000, ls = "--", marker = "o", markersize = 6, label = "$T = 50000$")
axes.loglog(l, l_gibbs_mh_tempn52_nenv1000_itgibbs10000,ls = "--", marker = "o", markersize = 6, label = "Gibbs $\\beta_n = n^{5/2}$")
axes.loglog(l, l_gibbs_mh_tempn2_env1000_itgibbs10000 , ls = "--", marker = "o", markersize = 6, label = "Gibbs $\\beta_n = n^2$")
#axes.loglog(l, l_gibbs_mh_tempn2_env5000_itgibbs10000 , ls = "--", marker = "o", markersize = 6, label = "MH $M_n = 5000$")
#axes.loglog(l, l_gibbs_mh_tempn2_envn_itgibbs10000, ls = "--", marker = "o", markersize = 6, label = "MH $M_n = n$")
axes.loglog(l_n, l_thinning, ls = "--", marker = "o", markersize = 6, label = "KT")
axes.loglog(l, l_gibbs_mh_temp_n3_env1000_itgibbs10000, ls = "--", marker = "o", markersize = 6, label = "Gibbs $\\beta_n = n^3$")
#axes.loglog(l, l_gibbs_coulomb_tempn2+c, ls = "--", marker = "o", markersize = 6, label = "$\\beta_n = n^2$")
#axes.loglog(l, l_gibbs_coulomb_tempn3+c, ls = "--", marker = "o", markersize = 6, label = "$\\beta_n = n^3$")

axes.legend()
axes.set_xlabel("Number of particles")
#axes.set_ylabel("$\\log I_{K}(\\mu_n - \\pi)$")
#axes.set_xscale("log")
#axes.set_yscale("log")
plt.grid(True, which="both", ls="-", color='0.65')
plt.tight_layout()
fig.savefig("comparison_variance.pdf")