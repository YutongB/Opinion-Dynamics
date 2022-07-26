
import numpy as np
from numpy.random import uniform

from src.simulation.sim import normalize_distr, prior_distr_vec

def parse_prior_str(s: str | None, n: int):
    # n is number of nodes, s is the str
    # examples:
    #   0.5 -> all nodes have 0.5 mean
    #   0.5,0.4,0.3 -> first node with 0.5 mean, second node with 0.4 mean, rest with 0.3 mean
    #   r -> all nodes have random mean
    #   0.5,r -> first node is 0.5 mean, the rest are random

    if s == None: # make a completely random prior
        return ['r'] * n

    s = [x if x == 'r' else float(x) for x in s.split(",") if x != '']
    # the rest have the same mean as the last one.
    rest = [s[-1]] * (n-len(s))
    s += rest
    return s

from numpy.random import uniform

def gen_prior_param(s: str, n: int, range: tuple[int, int]):
    rand = uniform(*range, n)  # n random priors within the given range
    params = parse_prior_str(s, n)
    # replace each r with a random value
    params = [r if m == 'r' else m for m, r in zip(params, rand)]
    return params
    
def gen_prior(mean: str | None, sd: str | None, n: int, mean_range, sd_range):
    prior_mean = gen_prior_param(mean, n, mean_range)
    prior_sd = gen_prior_param(sd, n, sd_range)
    return prior_mean, prior_sd


def init_simulation(g, prior):

    mean, sd, n, mean_range, sd_range = prior['mean'], prior['sd'], prior['n'], prior['mean_range'], prior['sd_range']
    
    graph_n = g.num_vertices(ignore_filter=True)
    assert n == graph_n, f"Prior n mismatches graph n ({n}, {graph_n})"

    prior_mean, prior_sd = gen_prior(mean, sd, n, mean_range, sd_range)
    assert len(prior_mean) == len(prior_sd) and len(prior_mean) == n, \
        f"Generated prior n mismatches graph n ({n}, {len(prior_mean)}, {len(prior_sd)})"

    prior_mean, prior_sd = np.array(prior_mean), np.array(prior_sd)

    # equivalent to g.vp.prior_mean.get_array()[:]
    g.vp.prior_mean.a = prior_mean.T
    g.vp.prior_sd.a = prior_sd.T

    distr = prior_distr_vec(n, prior_mean, prior_sd)
    distr = normalize_distr(distr)

    # required for indirect array access of vector
    # g.vp.prior_distr.set_2d_array(distr.T)
    return distr
