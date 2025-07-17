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

d = 3

def norm_2_safe_for_grad(x) :
      return jnp.power(jnp.linalg.norm(jnp.where(x != 0., x, 0.)), 2)
def K_riesz(x, y, eps = 0.1, s = d-2) : #coulomb
    return jnp.power(norm_2_safe_for_grad(x-y) + eps**2, -s/2)

#define test function
key = random.PRNGKey(0)
key_1, key_2, _ = random.split(key, 3)
unif = numpyro.distributions.Uniform()
z = unif.sample(key_1, sample_shape=(d, 10 ))
a = unif.sample(key_2, sample_shape=(1, 10 ))
def f(x):
    K = jnp.array([a[:, i] * K_riesz(z[:, i], x) for i in range(10)])
    return jnp.sum(K)


#import args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that specifies parameters"
    )
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--file_prefix", required=True, type=str)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--name_in_dic", required=True, type=str)
    args = parser.parse_args()
    dir = args.dir
    file_prefix = args.file_prefix
    n = args.n
    name_in_dic = args.name_in_dic

#import pickles from args n and dir and prefix in file names
samples_s = jnp.zeros(shape = (100, d, n))
s = file_prefix+str(n)+"_"
i = 0
#dir =  "pickles_gibbs_mh_tempn52_nenv1000_itgibbs10000"
for file in os.listdir(dir):
    if file.startswith(s):
        print(file)
        dic_file = pickle.load(open(dir + "/"+file,  "rb"))
        samples_s = samples_s.at[i, :, :].set(dic_file[name_in_dic])
        i+=1

def linear_stat(sample):
    #sample should have shape (d, n)
    #compute 1/n sum f(x_i)
    return jnp.mean(jit(vmap(f))(sample.T), axis = 0)

def empirical_variance(samples):
    #samples should have shape (100, d, n)
    #compute var(1/n sum f(x_i))
    L_linear_stat = jnp.array([linear_stat(samples[i, :,  :]) for i in range(samples.shape[0])])
    var_linear_stat =  jnp.var(L_linear_stat, axis = 0)
    return var_linear_stat


l_var = pickle.load(open("var.p", "rb"))
l_var_dir = l_var.get(dir, None)
if l_var_dir == None:
    l_var[dir] = {}
    l_var_dir = l_var[dir]
var = empirical_variance(samples_s)
l_var_dir[str(n)] = var
print(l_var_dir)
l_var[dir] = l_var_dir

pickle.dump(l_var, open("var.p", "wb"))