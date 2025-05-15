from functools import cache
import numpy as np

BIAS_SAMPLES = 21

@cache
def bias():
    # discretise the continuous variable theta (bias) into 21 regularly-spaced
    # values  (0.00, 0.05, ..., 1.00)
    return np.linspace(0, 1, BIAS_SAMPLES)

@cache
def bias_mat(num_priors):
    """
    generate matrix of num_prior rows and
    21 columns (bias() for each row of the matrix)
    """
    return np.tile(bias(), (num_priors, 1))


def normalize_distr(distr):
    return distr / np.sum(distr, axis=1)[:, None]
