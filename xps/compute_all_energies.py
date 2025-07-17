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
jax.config.update("jax_enable_x64", True)
import pickle
import numpy as np
import seaborn as sns

l_n = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
l_n_thinning =[50, 100, 150, 200, 250, 300, 350] 

#import args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that specifies parameters"
    )
    parser.add_argument("--dir", required=True, type=str)
    args = parser.parse_args()
    dir = args.dir

dirs ={'pickles_gibbs_mh_tempn2_env1000_itgibbs10000' : {'dir' : 'pickles_gibbs_mh_tempn2_env1000_itgibbs10000', 'file_prefix': 'points_gibbs_mh_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'},
        'pickles_gibbs_mh_tempn2_env5000_itgibbs10000' : {'dir' : 'pickles_gibbs_mh_tempn2_env5000_itgibbs10000', 'file_prefix': 'points_gibbs_mh_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'},
        'pickles_gibbs_mh_tempn2_envn_itgibbs10000' : {'dir' : 'pickles_gibbs_mh_tempn2_envn_itgibbs10000', 'file_prefix': 'points_gibbs_mh_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'},
        'pickles_gibbs_mh_tempn2_nenv1000_itgibbs50000' : {'dir' : 'pickles_gibbs_mh_tempn2_nenv1000_itgibbs50000', 'file_prefix': 'points_gibbs_mh_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'},
        'pickles_thinning' : {'dir' : 'pickles_thinning', 'file_prefix': 'points_thinning_', 'l_n' : l_n_thinning, 'name_in_dic' : 'points_thinned'},
        'pickles_gibbs_mh_tempn52_nenv1000_itgibbs10000' : {'dir' : 'pickles_gibbs_mh_tempn52_nenv1000_itgibbs10000', 'file_prefix': 'points_gibbs_mh_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'},
        'pickles_gibbs_mh_temp_n3_env1000_itgibbs10000' : {'dir' : 'pickles_gibbs_mh_temp_n3_env1000_itgibbs10000', 'file_prefix': 'points_gibbs_mh_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'},
        'pickles_mcmc' : {'dir' : 'pickles_mcmc', 'file_prefix': 'points_mcmc_', 'l_n' : l_n, 'name_in_dic' : 'points_mcmc'},
        'pickles_gibbs_coulomb_tempn2': {'dir' : 'pickles_gibbs_coulomb_tempn2', 'file_prefix': 'points_gibbs_coulomb_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'},
        'pickles_gibbs_coulomb_tempn3': {'dir' : 'pickles_gibbs_coulomb_tempn3', 'file_prefix': 'points_gibbs_coulomb_', 'l_n' : l_n, 'name_in_dic' : 'points_gibbs'}}

#l_energies = {}
#pickle.dump(l_energies, open("energies.p", "wb"))

args_dir = dirs[dir]['dir']
file_prefix = dirs[dir]['file_prefix']
l_n = dirs[dir]['l_n']
name_in_dic = dirs[dir]['name_in_dic']

for n in l_n:
    os.system("python3 compute_energies.py --dir="+args_dir+" --file_prefix="+file_prefix+" --n="+str(n)+" --name_in_dic="+name_in_dic)

