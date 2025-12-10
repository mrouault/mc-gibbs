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

paths = ["gibbs_mala_0_0_0.001_0.0001_1000_10000_100_1000000.0.p",
        "gibbs_mala_0_0_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mala_0_1_0.001_0.0001_1000_10000_100_1000000.0.p",
        "gibbs_mala_0_1_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mala_0_2_0.001_0.0001_1000_10000_100_1000000.0.p",
        "gibbs_mala_0_2_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mala_0_3_0.001_0.0001_1000_10000_100_1000000.0.p",
        "gibbs_mala_0_3_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mala_0_4_0.001_0.0001_1000_10000_100_1000000.0.p",
        "gibbs_mala_0_4_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mh_0_0_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mh_0_0_0.001_1e-05_1000_10000_100_1000000.0.p",
        "gibbs_mh_0_1_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mh_0_1_0.001_1e-05_1000_10000_100_1000000.0.p",
        "gibbs_mh_0_2_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mh_0_2_0.001_1e-05_1000_10000_100_1000000.0.p",
        "gibbs_mh_0_3_0.001_1e-05_1000_10000_100_10000.0.p",
        "gibbs_mh_0_3_0.001_1e-05_1000_10000_100_1000000.0.p",
        "gibbs_mh_0_4_0.001_0.0001_1000_10000_100_10000.0.p",
        "gibbs_mh_0_4_0.001_1e-05_1000_10000_100_1000000.0.p"]

paths_2 = ["gibbs_mh_0_3_0.001_0.0001_1000_10000_100_10000.0.p"]

for s in paths_2:
    print(s)
    dic = pickle.load(open(s, "rb"))
    points_gibbs = dic["points_gibbs"]
    points_gibbs_last = {}
    for k in range(100):
        points_gibbs_last[k] = points_gibbs[k][-1, :, :] #shape (d, n), picked only last iteration
    dic["points_gibbs"] = points_gibbs_last
    print(points_gibbs_last[99].shape)
    pickle.dump(dic, open("last_"+s, "wb"))
