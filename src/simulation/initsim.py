
import numpy as np
from numpy.random import uniform

from src.simulation.sim import fwhm_to_sd, normalize_distr, prior_distr_vec


def generate_priors(num_priors, mean_range=(0.0, 1.0), fwhm_range=(0.2, 0.8)):
    prior_mean = uniform(*mean_range, num_priors)
    prior_fwhm = uniform(*fwhm_range, num_priors)
    prior_sd = fwhm_to_sd(prior_fwhm)

    return prior_mean, prior_sd

def init_simulation(g, prior_mean=None, prior_sd=None):
    n = g.num_vertices(ignore_filter=True)

    if prior_mean is not None:
        prior_mean = np.array(prior_mean)
        if prior_mean.shape[0] != n:
            raise Exception("Invalid prior mean entered, g has dimension", n)
    if prior_sd is not None:
        prior_sd = np.array(prior_sd)
        if prior_sd.shape[0] != n:
            raise Exception("Invalid prior sd entered, g has dimension", n)

    # randomly generate prior and prior sd if no args given
    if prior_mean is None or prior_sd is None:
        prior_mean, prior_sd = generate_priors(n)

    # equivalent to g.vp.prior_mean.get_array()[:]
    g.vp.prior_mean.a = prior_mean.T
    g.vp.prior_sd.a = prior_sd.T

    # transpose is O(1) !!!
    distr = prior_distr_vec(n, prior_mean, prior_sd)
    distr = normalize_distr(distr)

    # required for indirect array access of vector
    # g.vp.prior_distr.set_2d_array(distr.T)
    return distr
