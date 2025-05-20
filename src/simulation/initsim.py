
import numpy as np
from scipy.stats import norm

from src.simulation.stat import normalize_distr

def prior_distr_vec(prior_mean, prior_sd, bias_mat: np.array):
    # ensure norm.pdf behaves (broadcasting - scipy)
    assert len(prior_mean) == len(prior_sd)
    num_priors = len(prior_mean)
    prior_mean.shape = prior_sd.shape = (num_priors, 1)
    return norm.pdf(bias_mat, loc=prior_mean, scale=prior_sd)


def init_prior_distrs(prior_mean: np.array, prior_sd: np.array, bias_mat) -> np.array:
    prior_mean, prior_sd = np.array(prior_mean), np.array(prior_sd)
    return normalize_distr(prior_distr_vec(prior_mean, prior_sd, bias_mat))


def prior_distr(prior_mean, prior_sd, bias: np.array):
    return norm.pdf(bias, loc=prior_mean, scale=prior_sd)


