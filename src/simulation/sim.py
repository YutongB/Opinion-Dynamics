from collections import namedtuple
from functools import cache
from scipy.stats import norm, binom
from numpy.random import uniform
import numpy as np
import graph_tool.all as gt

BIAS_SAMPLES = 21


def toss_coins(bias=0.5, num_coins=1):
    # heads=1 (success is for the coin to land heads), tails=0
    return np.random.binomial(num_coins, bias)
    

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
    return np.tile(np.linspace(0, 1, BIAS_SAMPLES), (num_priors, 1))



def prior_distr(prior_mean, prior_sd):
    return norm.pdf(bias(), loc=prior_mean, scale=prior_sd)


def prior_distr_vec(num_priors, prior_mean, prior_sd):
    # ensure norm.pdf behaves (broadcasting - scipy)
    prior_mean.shape = prior_sd.shape = (num_priors, 1)
    return norm.pdf(bias_mat(num_priors), loc=prior_mean, scale=prior_sd)


def normalize_distr(distr):
    norm = np.sum(distr, axis=1)
    return distr / norm[:, None]


@cache
def coin_toss_likelihood(num_heads, num_coins=1):
    """
    coin toss likelihood - encodes the probability of observing a heads, given bias θ

    For 1 coin toss: only 2 possible outcomes in a coin toss:
    row 0 (tails) : 1 - θ
    row 1 (heads) : θ 
    """
    return binom.pmf(num_heads, num_coins, bias())


def mean_distr(distrs):
    n = distrs.shape[0]
    return np.sum(distrs * bias_mat(n), axis=1)


def std_distr(distrs, mean):
    n = distrs.shape[0]
    mean = mean[:, None]
    return np.sqrt(
        np.sum(np.square(bias_mat(n) - mean) * distrs, axis=1))


def friendliness_mat(g):
    # takes roughly 300ms for graph of n=10
    return gt.adjacency(g, weight=g.ep.friendliness).toarray()


def adjacency_mat(g):
    return gt.adjacency(g).toarray()


def avg_dist_in_belief(friendliness, posterior_distr):
    n = friendliness.shape[0]

    xi = np.broadcast_to(
        posterior_distr, (n, n, BIAS_SAMPLES)).transpose((1, 0, 2))
    xj = np.broadcast_to(posterior_distr, (n, n, BIAS_SAMPLES))
    # calculate the difference in belief for node i to node j
    diff_in_belief = ((xj - xi).T * friendliness).T
    summation = np.sum(diff_in_belief, axis=1)
    # sums over the columns
    divisor = np.reciprocal(np.sum(np.abs(friendliness), axis=1))

    # the .T stuff fixes some broadcasting issue  (same above too!)
    avg_dist_in_belief = summation * divisor[:, None]
    return avg_dist_in_belief


def DW_update(friendliness, prior_distr, DWeps):
    # NOTE: DWeps = 1 makes this function do nothing.
    if DWeps == 1:
        return friendliness
    n = friendliness.shape[0]

    # if |prior x - prior y| > DWesp and friendliness[x][y] != 0, then ignore and friendliness[x][y] = 0
    xi = np.broadcast_to(prior_distr, (n, n, BIAS_SAMPLES)
                         ).transpose((1, 0, 2))
    xj = np.broadcast_to(prior_distr, (n, n, BIAS_SAMPLES))

    # calculate the difference in belief for node i to node j
    f = friendliness != 0
    # here, diff_in_belief is a distribution for each node i and node j
    diff_in_belief = np.abs(((xj - xi).T * f).T)   # shape: (n, n, BIAS_SAMPLES)
    # if any point within the difference in belief distribution is > DWeps, we will set
    # the friendliness of that to zero.
    mask = np.any(diff_in_belief <= DWeps, axis=2)
    if np.any(~mask):
        print("DEBUG: friend removed!","friend relationship removed = ", (~mask).sum()//2)
        
    friendliness = friendliness * mask

    return friendliness


EPSILON = 0


def step_simulation(g, prior_distr, true_bias=0.5, learning_rate=0.25, num_coins=1,
                    friendliness=None, DWeps=1, coins=None, disruption = 0):
    """
    true_bias (θ_0 in the paper)
    learning_rate (μ / μ_i in the paper)
    """
    n = g.num_vertices(ignore_filter=True)

    if friendliness is None:
        friendliness = friendliness_mat(g)

    if DWeps != 1:  # use the DW update rule
        friendliness = DW_update(friendliness, prior_distr, DWeps)

    if coins is None:
        # simulate an independent coin toss
        coins = toss_coins(bias=true_bias, num_coins=num_coins)
    # else: use the coin from the coins list.

    # update the opinions of each agent according to Bayes' Theorem (observe coin toss)
    # prior_distr1 = graph_get_prior_distr(g)
    likelihood = coin_toss_likelihood(
        num_heads=coins, num_coins=num_coins)  # (eqn 2)
    posterior_distr = normalize_distr(likelihood * prior_distr)  # (eqn 1)
    # Rows: node i's posterior distribution.  Cols: posterior distr evaluated at theta

    # Mix the opinions of each node with respective neighbours (eq 3,4)
    avg_dist_belief = avg_dist_in_belief(
        friendliness, posterior_distr)   # (eqn 4)

    # TODO: learning_rate vector, length # nodes; set a learning rate for each person
    bayes_update_max_RHS = posterior_distr + avg_dist_belief * learning_rate
    bayes_update_max_LHS = np.broadcast_to(EPSILON, (n, 21))
    bayes_update = np.amax(
        np.array([bayes_update_max_LHS, bayes_update_max_RHS]), axis=0)

    next_prior_distr = normalize_distr(bayes_update)
    next_prior_distr[:disruption] = prior_distr[:disruption]
    # graph_set_prior_distr(g, next_prior_distr.T)

    return coins, next_prior_distr


