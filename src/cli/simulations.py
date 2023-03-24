
import multiprocessing
from statistics import mean, stdev
from time import sleep
import numpy as np
import timeit
import json
from collections import defaultdict

from src.cli.make_sim_params import first_k_with_value_then_random
# from src.analyse.analyse import frac_asymptotic_system, num_asymptotic_agents, num_asymptotic_agents_theta
from src.simulation.graphs import gen_bba_graph, get_edge_generator, gen_complete_graph
from src.simulation.runsim import run_ensemble, run_simulation, make_progress
from src.simulation.initsim import gen_prior_param
from src.utils import timestamp
from graph_tool.topology import shortest_distance
from graph_tool import Graph

import logging

# use logger for thread-safe file writing (otherwise they might overwrite each other)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logpath = f"output/bigexperiment.log"
ch = logging.FileHandler(logpath)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

n = 100
runs = 100

bias = 0.6           # theta0
partisan_mean = 0.3  # thetap
partisan_sd = 0.01

asymp_max_iters = 99
max_steps = 10000

ret = {}

def make_coin_list(bias, max_steps, num_coins=1):
    return [np.random.binomial(num_coins, bias) for _ in range(max_steps)]

coinslist = make_coin_list(bias=bias, max_steps=max_steps)

def first_k_with_value_then_random(value, k):
    # eg: value = 5, k = 3 => "5,5,5,5,5,r" (the rest are treated as random, if any.)
    return ",".join([str(value)] * k) + ',r'

def flist_to_str(lst: list[float]):
    return ','.join([str(x) for x in lst])

# gen_graph = lambda: gen_complete_graph(n, get_edge_generator("enemies"))
# This is not working atm : int64 is not JSON serializable
def dwell_time_all_agents(asymptotic: list[int]) -> list[int]:
    steps = np.array(asymptotic)
    # last index at dwell
    dwell_indices = np.where(np.diff(steps) < 0)[0].tolist()
    dwell_times = [int(asymptotic[i]) for i in dwell_indices]
    if steps[-1] > 0:
        dwell_times.append(int(asymptotic[-1]))
    return dwell_times

def dwell_time_per_agent(asymptotic: list[list[int]]) -> list[list[int]]:
    """input: asymptotic - per agent, list of number of steps spent in asymptotic learning at time t
    output: dwell time - per agent, all maxima of asymptotic
    """
    return [dwell_time_all_agents(agent) for agent in asymptotic]

def dwell_time_per_agent_by_distance_to_partisan(
    graph: Graph, 
    num_partisans: int, 
    agent_is_asymptotic: list[list[int]]):

    # NOTE: the following assumes we do not find shortest path based on negative edge weights
    # this needs to change otherwise!
    for i, u in zip(range(num_partisans), graph.iter_vertices()):
        for _, v in zip(range(i + 1, num_partisans), graph.iter_vertices()):
            e = graph.add_edge(u, v)
            graph.ep.friendliness[e] = 0

    dist = shortest_distance(graph, source=0, weights=graph.ep.friendliness)
    dist_per_node = dist.a.astype(int).tolist()

    dwell = dwell_time_per_agent(agent_is_asymptotic)

    max_dist = max(dist_per_node)
    dwells_by_distance = [[] for _ in range(max_dist + 1)]
    number_of_agents_by_distance = [0 for _ in range(max_dist + 1)]
    for dist, dwell in zip(dist_per_node, dwell):
        dwells_by_distance[dist].extend(dwell)
        number_of_agents_by_distance[dist] += 1
    
    parts = zip(number_of_agents_by_distance, dwells_by_distance)
    return list(parts)

    # Agverage each sim
    # res = []
    # for n, d in parts:
    #     m = mean(d)
    #     s = stdev(d, m) if n > 1 else 0
    #     res.append((n, m, s))
    # return res

def asymptotic_per_agent(dists):
    steps = len(dists) - 1
    n = len(dists[0])
    asymptotic = np.zeros((n, steps), dtype=int)
    row = np.zeros(n, dtype=int)
    for i, (prior, posterior) in enumerate(zip(dists, dists[1:])):
        largest_change = np.max(np.abs(posterior - prior), axis=1)
        largest_peak = 0.01 * np.max(prior, axis=1)
        agent_is_asymptotic = largest_change < largest_peak  # one element for each agent
        row = np.where(agent_is_asymptotic, row + 1, 0)
        asymptotic[:, i] = row
        prior = posterior
    return asymptotic

def run(params):
    vals, run = params
    num_partisans, = vals
    m = 20
    gen_graph = lambda: gen_bba_graph(n, m=m, edge_generator=get_edge_generator("allies"))

    frac_partisans = num_partisans / n
    prior_mean = first_k_with_value_then_random(partisan_mean, num_partisans)
    prior_sd = first_k_with_value_then_random(partisan_sd, num_partisans)

    prior_mean = flist_to_str(gen_prior_param(prior_mean, n, range=(0,1)))
    prior_sd = flist_to_str(gen_prior_param(prior_sd, n, range=(0.2, 0.8)))

    sim_params = {
        "prior": {
            "mean": prior_mean,
            "sd": prior_sd,
            "n": n,
            "mean_range": (0,1),
            "sd_range": (0.2, 0.8),
        },
        "max_steps": max_steps,
        "true_bias": bias,
        "tosses_per_iteration": 1,
        "learning_rate": 0.25,
        "asymptotic_learning_max_iters": 99,
        "DWeps": 1,
        "disruption": num_partisans,
        "log": True,  # need to log to get mean_list
        "coinslist": None,  # coin list needs to be None to have different simulations
        "break_on_asymptotic_learning": False,
    }
    
    graph = gen_graph()
    sim = run_simulation(graph, **sim_params, progress=False)
    if sim is None:
        return None

    # use mean[num_partisans:] for all non-partisans
    # non_partisans_mean = [mean[num_partisans:].tolist() for mean in sim.mean_list]

    """
    agent_t_A = sim.steps - sim.agent_is_asymptotic
    agent_final_mean = sim.mean_list[-1]

    right_wrong_mask = np.isclose(agent_final_mean, bias, atol=0.001)
    t_right = agent_t_A[right_wrong_mask]
    t_wrong = agent_t_A[~right_wrong_mask]

    mean_t_right = np.mean(t_right)
    mean_t_wrong = np.mean(t_wrong)
    mean_t_diff = mean_t_right - mean_t_wrong
    # out = dict(partisans=round(frac_partisans, 2), mean_t_diff=mean_t_diff, run=run)
    # out = dict(m=m, mean_t_diff=mean_t_diff, run=run)
    """
    asymptotic = asymptotic_per_agent(sim.distrs)
    res = dwell_time_per_agent_by_distance_to_partisan(graph, num_partisans, asymptotic)
    out = dict(
        r=run, 
        p=round(frac_partisans, 2), 
        d=res,
    )

    logger.info(json.dumps(out))
        
    return params

def task_name(vals):
    return f"npartisans={vals[0]}"  # , m={vals[1]}"

def do_multithreaded(threads: int, param_sweep: list[tuple[list, int]], progress):
    sweep = progress.add_task("[bold purple]Sweeping", total=len(param_sweep), )
    runs_completed = defaultdict(lambda: 0)
    task_ids = defaultdict(lambda: None)

    with multiprocessing.Pool(threads) as pool:
        for res in pool.imap_unordered(run, param_sweep):
            vals, i = res
            if task_ids[vals] is None:
                task_ids[vals] = progress.add_task(task_name(vals), total=runs)
            progress.advance(task_ids[vals])
            progress.advance(sweep)
            runs_completed[vals] += 1
            if runs_completed[vals] == runs:
                progress.remove_task(task_ids[vals])
                task_ids[vals] = None

# I don't know why but logger refuse to work with single threaded
def do_singlethreaded(param_sweep: list[tuple[list, int]], progress):
    sweep = progress.add_task("[bold purple]Sweeping", total=len(param_sweep), )
    runs_completed = defaultdict(lambda: 0)
    task_ids = defaultdict(lambda: None)

    for vals, i in param_sweep:
        if task_ids[vals] is None:
            task_ids[vals] = progress.add_task(task_name(vals), total=runs)
        run((vals, i))
        progress.advance(task_ids[vals])
        progress.advance(sweep)
        runs_completed[vals] += 1
        if runs_completed[vals] == runs:
            progress.remove_task(task_ids[vals])
            task_ids[vals] = None

def main():
    import argparse

    print("will write to log file: ", logpath)
    this_log = logpath.split(".")[0]
    this_logpath = this_log + "-" + timestamp() + ".log"
    print("final logfile: ", this_logpath)

    # select number of threads
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    time1 = timeit.default_timer()

    # delete log file if it exists
    try:
        import os
        os.remove(logpath)
    except OSError:
        pass

    param_sweep = [(x,) for x in range(1, n)]
    
    with make_progress() as progress:
        param_sweep = [(vals, run) for vals in param_sweep for run in range(runs)]
        # if args.threads > 1:
        do_multithreaded(args.threads, param_sweep, progress)
        # else:
            # do_singlethreaded(param_sweep, progress)
        
    time2 = timeit.default_timer()

    import shutil
    shutil.copyfile(logpath, this_logpath)
    print("Wrote log to ", this_logpath)
    print(f'Time taken: {(time2 - time1):.2f} seconds')

if __name__ == '__main__':
    main()