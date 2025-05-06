from typing import Callable
from jax import numpy as jnp
from jax import scipy
from jax import random, Array, jit, vmap, grad
from jax.tree_util import Partial as partial
from jax.lax import fori_loop, cond, dynamic_slice
import numpyro
import optax
import jax
from mcmc_samplers import mh, mala

class gibbs(numpyro.distributions.Distribution) :

    def __init__(self, d, n, K, beta_n, V) :
        '''
        :param K : translation invariant kernel function K : R^d x R^d -> R, might be infinite on the diagonal.
        '''

        self.n = n
        self.K = K
        self.V = V
        self.d = d
        self.beta_n = beta_n
        event_shape = (n, d)
        super(gibbs, self).__init__(event_shape = event_shape)


    def log_prob(self, value) :

        #value = (x_1, \dots, x_n)
        value_shape = value.shape #should be (1, n*d)
        X = jnp.array(value.reshape(self.d, self.n))
        index = jnp.array([k for k in range(self.n)])

        k_inter = lambda j ,k : jnp.where(j == k, 0., self.K(X[:, j], X[:, k])) #doesn't count diagonal terms
        pair_j = lambda j : jit(vmap(partial(k_inter, j)))(index).sum()
        pair_jit = jit(vmap(pair_j))(index).sum()
        pair_inter = self.beta_n * (pair_jit )/  (2*self.n**2)

        v_inter = lambda j : self.V(X[:, j])
        ext_jit = jit(vmap(v_inter))(index).sum()
        exter_V = self.beta_n * ext_jit / self.n

        return - pair_inter - exter_V

    def sample(self, key, start_sample, n_iter, step_size, method = 'mala') :
        '''
        :start_sample : Array with the first element with shape (1, n*d).
        return : Dic with last sample of the chain with shape (1, n*d) and the acceptance rate.
        Sample should be reshaped as (d, n).T to get the shape (n, d).
        '''
        start_sample = jnp.atleast_2d(start_sample)
        target_mala = lambda x : self.log_prob(x)
        sample_mala, acceptance = jit(vmap(partial(mala,
                               log_prob_target = target_mala,
                               n_iter = n_iter,
                               step_size =  step_size)))(random.split(key, 1), start_sample)
        return {"samples": sample_mala, "acceptance": acceptance}