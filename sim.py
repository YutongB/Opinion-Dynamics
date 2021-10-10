from collections import namedtuple
from functools import cache
from operator import pos
from scipy.stats import norm, binom
from numpy.random import seed, uniform
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt
from random import choice
from datetime import datetime

import json
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

import signal
from threading import Event
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

def make_progress():
    return Progress(
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=15),
        "[progress.percentage]{task.completed}",
        "/",
        "{task.total}",
        "•",
        TimeElapsedColumn(),
        "/",
        TimeRemainingColumn(),
        refresh_per_second=5,
        # transient=True,
    )

def create_model_graph(g=None):
    if g is None:
        g = gt.Graph(directed=False)
    # nodes represent people
    # edges represent 'knowing' relationships of people

    # friendliness:  # or g.ep.friendliness
    #   -1 if people are enemies
    #   +1 if people are friends'
    g.edge_properties["friendliness"] = g.new_edge_property("double")

    # each person has a prior distr, mean and std
    # or g.vp.prior_mean and g.vp.prior_sd
    g.vertex_properties["prior_mean"] = g.new_vertex_property("double")
    g.vertex_properties["prior_sd"] = g.new_vertex_property("double")
    g.vertex_properties["prior_distr"] = g.new_vertex_property(
        "vector<double>")

    # g.graph_properties['step'] = g.new_graph_property('int')

    return g


def create_random_graph():
    pass
    # TODO: generate complete graph
    # https://graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.random_graph


def draw_graph(g, show_vertex_labels=False, width=150):

    # edge_color_map = {-1: (1, 0, 0, 1), 1: (0, 1, 0, 1), 0: (0, 0, 0, 0)}
    # edge_colors = [edge_color_map[int(e)] for e in g.ep.friendliness]

    gt.graph_draw(g,
                  vertex_text=g.vertex_index if show_vertex_labels else None,
                #   edge_color=edge_colors,
                  edge_text=g.ep.friendliness,
                  output_size=(width, width))


ADJ_FRIEND = 1
ADJ_ENEMY = -1


def add_relationship(g, v1, v2, friendliness):
    e = g.add_edge(v1, v2)  # they know each other

    g.ep.friendliness[e] = friendliness  # if they like each other


def add_friends(g, v1, v2):
    add_relationship(g, v1, v2, ADJ_FRIEND)


def add_enemies(g, v1, v2):
    add_relationship(g, v1, v2, ADJ_ENEMY)


def pair_of_allies():
    g = create_model_graph()
    v1, v2 = g.add_vertex(2)
    add_friends(g, v1, v2)
    return g


def pair_of_opponents():
    g = create_model_graph()
    v1, v2 = g.add_vertex(2)
    add_enemies(g, v1, v2)
    return g

# TODO: Random graph generation
# TODO: argv
# TODO: better progress bar


def gen_complete_graph(n, generator):
    g = create_model_graph()
    v = g.add_vertex()
    vlist = [v]

    for _i in range(1, n):
        u = g.add_vertex()
        for v in vlist:
            add_relationship(g, u, v, generator())
        vlist.append(u)

    return g


def gen_relationship_binary():
    return choice([ADJ_ENEMY, ADJ_FRIEND])


def gen_relationship_uniform():
    return uniform(ADJ_ENEMY, ADJ_FRIEND)


def complete_graph_of_friends(n):
    return gen_complete_graph(n, lambda: ADJ_FRIEND)


def complete_graph_of_enemies(n):
    return gen_complete_graph(n, lambda: ADJ_ENEMY)


def complete_graph_of_random(n):
    return gen_complete_graph(n, gen_relationship_binary)


def complete_graph_of_random_uniform(n):
    return gen_complete_graph(n, gen_relationship_uniform)


HEADS = 1  # "success" is for the coin to land heads
TAILS = 0

BIAS_SAMPLES = 21


def toss_coins(bias=0.5, num_coins=1):
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


def fwhm_to_sd(fwhm):
    # full width half mean
    # 2 * sqrt(2 * ln(2))  (see wiki) (don't calculate sqrt or ln)
    return fwhm / 2.3548200450309493820231386529


def gen_prior_mean(num_priors, mean_range=(0.0, 1.0)):
    return uniform(*mean_range, num_priors)

def gen_prior_sd(num_priors, fwhm_range=(0.2, 0.8), sd_range=None):
    if sd_range is not None:
         # alternatively generate directly from standard deviation
        return uniform(*sd_range, num_priors)
    return fwhm_to_sd(uniform(*fwhm_range, num_priors))

def generate_priors(num_priors, mean_range=(0.0, 1.0), fwhm_range=(0.2, 0.8)):
    prior_mean = uniform(*mean_range, num_priors)
    prior_fwhm = uniform(*fwhm_range, num_priors)
    prior_sd = fwhm_to_sd(prior_fwhm)

    return prior_mean, prior_sd


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
    # return np.array((1 - bias(), bias()))

def mean_distr(distrs):
    n = distrs.shape[0]
    return np.sum(distrs * bias_mat(n), axis=1)

def std_distr(distrs, mean):
    n = distrs.shape[0]
    mean = mean[:, None]
    return np.sqrt(
        np.sum(np.square(bias_mat(n) - mean) * distrs, axis=1))

def mean_std_distr(distrs):
    mean = mean_distr(distrs)
    std = std_distr(distrs, mean)
    return np.array([mean, std]).T


def print_mean_std(mean_std):
    for i, person in enumerate(mean_std):
        print(i, "mean=", person[0], "std=", person[1])


def plot_distr(distr, title=None):
    fig, ax = plt.subplots()

    for v, d in enumerate(distr):
        ax.plot(bias(), d, label=str(v))

    ax.set_xlabel(r'Bias, $\theta$')
    ax.set_ylabel('Probability')

    ax.grid(linestyle='--', axis='x')
    ax.set_title(title)
    # plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
    plt.tight_layout()


def plot_graph_distr(g):
    fig, ax = plt.subplots()

    print_mean_std(mean_std_distr(graph_get_prior_distr(g)))

    for v, distr in g.iter_vertices([g.vp.prior_distr]):
        ax.plot(bias(), distr, label=str(v))

    ax.set_xlabel(r'Bias, $\theta$')
    ax.set_ylabel('Probability')

    ax.grid(linestyle='--', axis='x')
    # plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
    plt.tight_layout()


def graph_get_prior_distr(g):
    """
    Returns matrix: 
        row i corresponding to the prior distribution for person i, x_i(t, θ)
        col j corresponding to the prior distribution evaluated at θ = 0.05 * j
    """
    return g.vp.prior_distr.get_2d_array(np.arange(21)).T


def graph_set_prior_distr(g, distr):
    """
    Set matrix: 
        row i corresponding to the prior distribution for person i, x_i(t, θ)
        col j corresponding to the prior distribution evaluated at θ = 0.05 * j
    """
    return g.vp.prior_distr.set_2d_array(distr.T)


def friendliness_mat(g):
    # takes roughly 300ms for graph of n=10
    return gt.adjacency(g, weight=g.ep.friendliness).toarray()


def avg_dist_in_belief(friendliness, posterior_distr):
    n = friendliness.shape[0]

    xi = np.broadcast_to(posterior_distr, (n, n, 21)).transpose((1, 0, 2))
    xj = np.broadcast_to(posterior_distr, (n, n, 21))
    # calculate the difference in belief for node i to node j
    diff_in_belief = ((xj - xi).T * friendliness).T
    summation = np.sum(diff_in_belief, axis=1)

    # sums over the columns
    divisor = np.reciprocal(np.sum(np.abs(friendliness), axis=1))

    # the .T stuff fixes some broadcasting issue  (same above too!)
    avg_dist_in_belief = summation * divisor[:, None]
    return avg_dist_in_belief


# mean_range=(0.0, 1.0), fwhm_range=(0.2, 0.8)
def init_simulation(g, prior_mean=None, prior_sd=None):
    n = g.num_vertices(ignore_filter=True)

    if prior_mean is not None and prior_mean.shape[0] != n:
        raise Exception("Invalid prior mean entered, g has dimension", n)
    if prior_sd is not None and prior_sd.shape[0] != n:
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

EPSILON = 0

def step_simulation(g, prior_distr, true_bias=0.5, learning_rate=0.25, num_coins=10, friendliness=None):
    """
    true_bias (θ_0 in the paper)
    learning_rate (μ / μ_i in the paper)
    """
    n = g.num_vertices(ignore_filter=True)

    if friendliness is None:
        friendliness = friendliness_mat(g)

    # simulate an independent coin toss
    toss = toss_coins(bias=true_bias, num_coins=num_coins)

    # update the opinions of each agent according to Bayes' Theorem (observe coin toss)
    # prior_distr1 = graph_get_prior_distr(g)
    likelihood = coin_toss_likelihood(
        num_heads=toss, num_coins=num_coins)  # (eqn 2)
    posterior_distr = normalize_distr(likelihood * prior_distr)  # (eqn 1)
    # Rows: node i's posterior distribution.  Cols: posterior distr evaluated at theta

    # Mix the opinions of each node with respective neighbours (eq 3,4)
    avg_dist_belief = avg_dist_in_belief(friendliness, posterior_distr)   # (eqn 4)

    # TODO: learning_rate vector, length # nodes; set a learning rate for each person

    bayes_update_max_RHS = posterior_distr + avg_dist_belief * learning_rate
    bayes_update_max_LHS = np.broadcast_to(EPSILON, (n, 21))
    bayes_update = np.amax(
        np.array([bayes_update_max_LHS, bayes_update_max_RHS]), axis=0)

    next_prior_distr = normalize_distr(bayes_update)

    # graph_set_prior_distr(g, next_prior_distr.T)

    # plot_graph_distr(g)

    return toss, next_prior_distr

done_event = Event()
def handle_sigint(signum, frame):
    done_event.set()
signal.signal(signal.SIGINT, handle_sigint)

SimResults = namedtuple("SimulationResults",
                        ("steps", "asymptotic", "coins", "mean_list", "std_list", 
                        "final_distr", "initial_distr", "friendliness", "adjacency"))

def adjacency_mat(g):
    return gt.adjacency(g).toarray()

def run_simulation(g, max_steps=1e4, asymptotic_learning_max_iters=10,
                   prior_mean=None, prior_sd=None,
                   true_bias=0.5, learning_rate=0.25, 
                   tosses_per_iteration=10, task_id=None, progress=None,
                   log=None):
    """
    max_steps (T in the paper) - maximum number of steps to run the simulation
    """
    initial_distr = init_simulation(g, prior_mean=prior_mean, prior_sd=prior_sd)

    coins_list = []
    mean_list = []
    std_list = []
    iters_asymptotic_learning = 0
    prior_distr = initial_distr.copy()

    max_steps = int(max_steps)

    # For now, static friendliness; TODO: update friendliness dynamically. (Hi Christine!)
    friendliness = friendliness_mat(g)

    if progress is None:
        progress = make_progress()

    if task_id is None:
        task_id = progress.add_task("Simulation", total=max_steps)
    else:
        progress.start_task(task_id)

    progress.update(task_id, total=max_steps)

    # steps 2-3 of Probabilistic automaton, until t = T
    for i in range(max_steps):
        coins, posterior = step_simulation(
            g, prior_distr=prior_distr, true_bias=true_bias, learning_rate=learning_rate, 
            friendliness=friendliness,
            num_coins=tosses_per_iteration)

        if log is not None:
            mean = mean_distr(posterior)
            mean_list.append(mean)
            std_list.append(std_distr(posterior, mean))
            coins_list.append(coins)

        if np.all(np.any(posterior > 0.99, axis=1)):
            iters_asymptotic_learning += 1
        else:
            iters_asymptotic_learning = 0
        
        progress.update(task_id, advance=1)

        if iters_asymptotic_learning == asymptotic_learning_max_iters:
            progress.update(task_id, total=i+1, completed=i+1)
            break

        prior_distr = posterior.copy()

        if done_event.is_set():
            print("!! Caught interrupt.")
            return

    adjacency = adjacency_mat(g)

    return SimResults(steps=len(coins_list),
                      asymptotic=iters_asymptotic_learning == asymptotic_learning_max_iters,
                      coins=coins_list,
                      mean_list=mean_list,
                      std_list=std_list,
                      final_distr=prior_distr,
                      initial_distr=initial_distr,
                      adjacency=adjacency,
                      friendliness=friendliness)


def do_ensemble(runs=1000, gen_graph=None, sim_params=None, simple=False):
    """
    sim_params: dictionary of parameters to pass to run_simulation
    eg: sim_params = { "max_steps": 100 }
    """

    # by default, do complete graph of 10 nodes with random
    if gen_graph is None:
        def gen_graph(): 
            return complete_graph_of_random(10)

    if sim_params is None:
        sim_params = {}


    with make_progress() as progress:
        results = []
        ensemble = progress.add_task("Ensemble", total=runs)

        for r in range(runs):
            task_id = progress.add_task("Sim #{}".format(r+1))
            sim = run_simulation(gen_graph(), **sim_params, task_id=task_id, progress=progress)
            
            #print("Run {}/{}: Asymptotic Learning Time: {}".format(r+1, runs, sim.steps))
            if simple:
                results.append(sim.step if sim.asymptotic else 0)
            else:
                results.append(sim)
            progress.update(ensemble, advance=1)


    return results


# TODO: This doesn't work because of the GIL
def do_ensemble_parallel(runs=1000, max_workers=10, gen_graph=None, sim_params=None):
    """
    sim_params: dictionary of parameters to pass to run_simulation
    eg: sim_params = { "max_steps": 100 }
    """
    raise NotImplemented("not working because of GIL")
    # by default, do complete graph of 10 nodes with random
    if gen_graph is None:
        def gen_graph(): return complete_graph_of_random(10)

    if sim_params is None:
        sim_params = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_simulation, gen_graph(), **sim_params) for r in range(runs)]

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    return results

# from dumping a numpy array to json : https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # list of ndarrays
        if isinstance(obj, list):
            if len(obj) != 0 and isinstance(obj[0], np.ndarray):
                return [el.tolist() for el in obj]
        return json.JSONEncoder.default(self, obj)


def dump_results(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        # convert namedtuple to dictionary
        json.dump([r._asdict() for r in results], f, cls=NumpyEncoder)

def dump_dict(d, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(d, f, cls=NumpyEncoder)

def parse_result(results_dict):
    for k in ["adjacency", "friendliness", "final_distr", "initial_distr", "mean_list", "std_list"]:
        results_dict[k] = np.asarray(results_dict[k])

    return SimResults(**results_dict)


def read_results(filename):
    with open(filename, 'r') as f:
        return [parse_result(r) for r in json.load(f)]
        #results = list(map(parse_result, results))

def matrix_to_edge_list(mat):
    n = len(mat)
    return [(u, v)
        for u in range(n) for v in range(u+1, n)]


def read_graph(adjacency, friendliness):
    g = create_model_graph()
    n = len(adjacency)
    g.add_vertex(n)
    g.add_edge_list([(u, v, friendliness[u][v]) 
        for u, v in matrix_to_edge_list(adjacency)], 
        eprops=[g.ep.friendliness])

    return g
    #friend_edges = {(u, v): friendliness[u][v] 
    #    for u in range(n) for v in range(u+1, n) }
    


def timestamp():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

