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
plt.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['text.usetex'] = True

file_mcmc = "pickles_mcmc/points_mcmc_500_0.p"
file_gibbs_n2_50k = "pickles_gibbs_mh_tempn2_nenv1000_itgibbs50000/points_gibbs_mh_500_0_tempn2_env1000_itgibbs50000.p"
file_gibbs_n3_50k = "pickles_gibbs_mh_temp_n3_env1000_itgibbs50000/points_gibbs_mh_500_0_tempn3_env1000_itgibbs50000.p"
file_gibbs_n3_100k = "pickles_gibbs_mh_temp_n3_env1000_itgibbs100000/points_gibbs_mh_500_0_tempn3_env1000_itgibbs100000.p"
file_gibbs_n3_200k = "pickles_gibbs_mh_temp_n3_env1000_itgibbs200000/points_gibbs_mh_500_0_tempn3_env1000_itgibbs200000.p"

points_mcmc = pickle.load(open(file_mcmc, "rb"))['points_mcmc']
points_gibbs_n2_50k = pickle.load(open(file_gibbs_n2_50k, "rb"))['points_gibbs']
points_gibbs_n3_50k = pickle.load(open(file_gibbs_n3_50k, "rb"))['points_gibbs']
points_gibbs_n3_100k = pickle.load(open(file_gibbs_n3_100k, "rb"))['points_gibbs']
points_gibbs_n3_200k = pickle.load(open(file_gibbs_n3_200k, "rb"))['points_gibbs']

fig, axes = plt.subplots()
axes.scatter(*points_mcmc[:2, :], alpha = 0.8, s = 10)
axes.set_aspect('equal',adjustable='box')
axes.set_xlim(-2.5, 2.5)
axes.set_ylim(-2.5, 2.5)
fig.savefig("points_mcmc.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_gibbs_n2_50k[:2, :], alpha = 0.8, s = 10)
axes.set_xlim(-2.5, 2.5)
axes.set_ylim(-2.5, 2.5)
axes.set_aspect('equal',adjustable='box')
fig.savefig("points_gibbs_n2_50k.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_gibbs_n3_50k[:2, :], alpha = 0.8, s = 10)
axes.set_aspect('equal',adjustable='box')
axes.set_xlim(-2.5, 2.5)
axes.set_ylim(-2.5, 2.5)
fig.savefig("points_gibbs_n3_50k.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_gibbs_n3_100k[:2, :], alpha = 0.8, s = 10)
axes.set_aspect('equal',adjustable='box')
axes.set_xlim(-2.5, 2.5)
axes.set_ylim(-2.5, 2.5)
fig.savefig("points_gibbs_n3_100k.pdf")

fig, axes = plt.subplots()
axes.scatter(*points_gibbs_n3_200k[:2, :], alpha = 0.8, s = 10)
axes.set_aspect('equal',adjustable='box')
axes.set_xlim(-2.5, 2.5)
axes.set_ylim(-2.5, 2.5)
fig.savefig("points_gibbs_n3_200k.pdf")
