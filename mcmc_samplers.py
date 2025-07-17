from typing import Callable
from jax import numpy as jnp
from jax import scipy
from jax import random, Array, jit, vmap, grad
from jax.tree_util import Partial as partial
from jax.lax import fori_loop, cond, dynamic_slice, scan
import numpyro
import optax
import jax

#-------------------------------------------------------
def mh(key: Array,
         start_sample: Array,
         log_prob_target: Callable[[jnp.ndarray], jnp.ndarray],
         n_iter: int,
         step_size: float) -> jnp.ndarray:
    '''
    This function implements MALA sampler.
    :param key: PRNGArray specifying the key that is used for the random numbers
    :param start_sample: Array with the first element with shape (n_chains, d) where d is the dimension of the target distribution and n_chains is the number of Markov chains.
    :param log_prob_target: Function that calculates the log probability of the target distribution at a given point.
    :param n_iter: int The number of steps of the chain.
    :param step_size: The step size for the Langevin kernel.
    :return: The samples along the chain and the acceptance rate computed online.
    '''
    start_sample = jnp.atleast_2d(start_sample)

    def mh_step(i, val):

        positions, key, acceptance = val
        sample = jnp.atleast_2d(positions[i-1, :])
        key, subkey_u, subkey_sample, _ = random.split(key, 4)
        #Sample the langevin kernel
        noise = random.normal(subkey_sample, (sample.shape[-1],))
        next = sample + jnp.sqrt(2 * step_size) * noise
        #Compute the log acceptance ratio
        fwd_ker_logprob = - (noise ** 2).sum() / 2
        bwd_ker_logprob = - ((sample - next)**2).sum() / (4 * step_size)
        log_ratio = log_prob_target(next) + bwd_ker_logprob - log_prob_target(sample) - fwd_ker_logprob

        log_u = jnp.log(random.uniform(subkey_u))
        accept = log_ratio > log_u
        x = jnp.where(log_ratio > log_u, next, sample)

        acceptance = (i*acceptance + accept)/(i+1)
        positions = positions.at[i, :].set(jnp.atleast_2d(x)[0, :])
        return (positions, key, acceptance)

    all_positions = jnp.zeros((n_iter, start_sample.shape[1]))
    all_positions.at[0, :].set(start_sample[0, :])

    positions, key, acceptance = fori_loop(1,
                          n_iter,
                          body_fun=mh_step,
                          init_val=(all_positions, key, jnp.array([0.])))
    return positions, acceptance

#-------------------------------------------------------
def mala(key: Array,
         start_sample: Array,
         log_prob_target: Callable[[jnp.ndarray], jnp.ndarray],
         n_iter: int,
         step_size: float) -> jnp.ndarray:
    '''
    This function implements MALA sampler.
    :param key: PRNGArray specifying the key that is used for the random numbers
    :param start_sample: Array with the first element with shape (1, d) where d is the dimension of the target distribution.
    :param log_prob_target: Function that calculates the log probability of the target distribution at a given point.
    :param n_iter: int The number of steps of the chain.
    :param step_size: The step size for the Langevin kernel.
    :return: The last sample of the chain and the acceptance rate computed online.
    '''
    start_sample = jnp.atleast_2d(start_sample)
    #you need to define the gradient of the log_prob_target parameter
    grad_logpdf = grad(lambda x: log_prob_target(x))

    def mh_step(i, val):

        sample, key, acceptance = val
        key, subkey_u, subkey_sample, _ = random.split(key, 4)
        #Sample the langevin kernel

        noise = random.normal(subkey_sample, (sample.shape[-1],))
        next = sample + step_size * grad_logpdf(sample) \
               + jnp.sqrt(2 * step_size) * noise
        #Compute the log acceptance ratio
        fwd_ker_logprob = - (noise ** 2).sum() / 2
        bwd_ker_logprob = - ((sample - next - step_size * grad_logpdf(next))**2).sum() / (4 * step_size)
        log_ratio = log_prob_target(next) + bwd_ker_logprob - log_prob_target(sample) - fwd_ker_logprob

        log_u = jnp.log(random.uniform(subkey_u))
        accept = log_ratio > log_u
        acceptance = (i*acceptance + accept)/(i+1)

        x = cond(accept,
                 lambda _: next,
                 lambda _: sample,
                 None)
        return (x, key, acceptance)

    sample, k, acceptance = fori_loop(0, n_iter, mh_step, (start_sample, key, jnp.array([0.])))
    sample = jnp.atleast_2d(sample)
    return sample, acceptance
