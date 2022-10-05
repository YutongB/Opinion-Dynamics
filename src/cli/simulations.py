
import multiprocessing
from time import sleep
import numpy as np
import timeit
import json

from src.cli.make_sim_params import first_k_with_value_then_random
from src.analyse.analyse import frac_asymptotic_system, num_asymptotic_agents, num_asymptotic_agents_theta
from src.simulation.graphs import gen_bba_graph, get_edge_generator, gen_complete_graph
from src.simulation.runsim import run_ensemble, run_simulation, make_progress
from src.simulation.initsim import gen_prior_param
from src.utils import timestamp

import logging

# use logger for thread-safe file writing (otherwise they might overwrite each other)
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

logpath = f"output/bigexperiment.log"
ch = logging.FileHandler(logpath)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

n = 100
runs = 1000
gap = 5
asymp_max_iters = 99
num_partisans_sweep = list(range(0, n, gap))
# num_partisans_sweep = [40]

bias = 0.6           # theta0
partisan_mean = 0.6  # thetap
partisan_sd = 0.01

max_steps = 10000

param_sweep = [(num_partisans, run) for num_partisans in num_partisans_sweep for run in range(runs)]

ret = {}

def make_coin_list(bias, max_steps, num_coins=1):
    return [np.random.binomial(num_coins, bias) for _ in range(max_steps)]

coinslist = make_coin_list(bias=bias, max_steps=max_steps)

def first_k_with_value_then_random(value, k):
    # eg: value = 5, k = 3 => "5,5,5,5,5,r" (the rest are treated as random, if any.)
    return ",".join([str(value)] * k) + ',r'

def flist_to_str(lst: list[float]):
    return ','.join([str(x) for x in lst])

gen_graph = lambda: gen_complete_graph(n, get_edge_generator("enemies"))
# gen_graph = lambda: gen_bba_graph(n, m=3, edge_generator=get_edge_generator("enemies"))

def run(params):
    num_partisans, run = params

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
        "break_on_asymptotic_learning": True,
    }
    
    sim = run_simulation(gen_graph(), **sim_params, progress=False)
    if sim is None:
        return None

    # use mean[num_partisans:] for all non-partisans
    # non_partisans_mean = [mean[num_partisans:].tolist() for mean in sim.mean_list]
    # asymptotic = sim.asymptotic

    agent_t_A = sim.steps - sim.agent_is_asymptotic
    agent_final_mean = sim.mean_list[-1]

    right_wrong_mask = np.isclose(agent_final_mean, bias, atol=0.001)
    t_right = agent_t_A[right_wrong_mask]
    t_wrong = agent_t_A[~right_wrong_mask]

    mean_t_right = np.mean(t_right)
    mean_t_wrong = np.mean(t_wrong)
    mean_t_diff = mean_t_right - mean_t_wrong

    out = dict(partisans=round(frac_partisans, 2), mean_t_diff=mean_t_diff, run=run)

    logger.info(json.dumps(out))
    return num_partisans


def main():
    time1 = timeit.default_timer()

    # delete log file if it exists
    try:
        import os
        os.remove(logpath)
    except OSError:
        pass
    
    with make_progress() as progress:

        sweep = progress.add_task("[bold purple]Sweeping", total=len(param_sweep), )
        runs_completed = [0] * len(num_partisans_sweep)
        task_ids = [None] * len(num_partisans_sweep)

        with multiprocessing.Pool(10) as pool:
            for res in pool.imap_unordered(run, param_sweep):

                if task_ids[res] is None:
                    task_ids[res] = progress.add_task(f"{res} partisans", total=runs)
                progress.advance(task_ids[res])
                progress.advance(sweep)
                runs_completed[res] += 1
                if runs_completed[res] == runs:
                    progress.remove_task(task_ids[res])
                    task_ids[res] = None

    time2 = timeit.default_timer()

    this_log = logpath.split(".")[0]
    with open(logpath, 'r') as f:
        data = f.read()
    this_logpath = this_log + "-" + timestamp() + ".log"
    with open(this_logpath, "w") as f:
        f.write(data)
    print("Wrote to", this_logpath)

    print(f'Time taken: {(time2 - time1):.2f} seconds')

if __name__ == '__main__':
    main()