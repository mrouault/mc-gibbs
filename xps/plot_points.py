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

file_mcmc = "pickles_mcmc/points_mcmc_500_0.p"
file_gibbs_n2 = "pickles_gibbs_mh_tempn2_env1000_itgibbs10000/points_gibbs_mh_500_0_tempn2_env1000_itgibbs10000.p"
file_gibbs_n3 = "pickles_gibbs_mh_temp_n3_env1000_itgibbs10000/points_gibbs_mh_500_0_tempn3_env1000_itgibbs50000.p"

points_mcmc = pickle.load(open(file_mcmc, "rb"))['points_mcmc']
points_gibbs_n2 = pickle.load(open(file_gibbs_n2, "rb"))['points_gibbs']
points_gibbs_n3 = pickle.load(open(file_gibbs_n3, "rb"))['points_gibbs']
points_thinning = pickle.load(open("pickles_thinning/points_thinning_350_0.p", "rb"))['points_thinned']
points_gibbs_coulomb_n2 = pickle.load(open("pickles_gibbs_coulomb_tempn2/points_gibbs_coulomb_500_0_tempn2_itgibbs10000_envn__tempenvn2_itgibbsenv10000.p", "rb"))['points_gibbs']

fig, axes = plt.subplots()
axes.scatter(*points_mcmc[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("points_mcmc.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_gibbs_n2[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("points_gibbs_n2.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_gibbs_n3[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("points_gibbs_n3.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_thinning[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("points_thinning.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_gibbs_coulomb_n2[:2, :], alpha = 0.8, s = 10)
axes.axis('equal')
fig.savefig("points_gibbs_coulomb_n2.pdf")