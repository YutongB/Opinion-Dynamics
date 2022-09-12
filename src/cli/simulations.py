
import multiprocessing
import numpy as np
import timeit
import json

from src.cli.make_sim_params import first_k_with_value_then_random
from src.analyse.analyse import frac_asymptotic_system, num_asymptotic_agents, num_asymptotic_agents_theta
from src.simulation.graphs import make_graph_generator
from src.simulation.runsim import run_ensemble
from src.simulation.initsim import gen_prior_param
from src.utils import timestamp

import logging

# use logger for thread-safe file writing (otherwise they might overwrite each other)
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

logpath = f"output/bigexperiment-1.log"
ch = logging.FileHandler(logpath)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)


n, runs = 100, 10
# n, runs = 100, 10
asymp_max_iters = 99
num_partisans_sweep = range(0, n)
# num_partisans_sweep = [40]


bias = 0.6           # theta0
partisan_mean = 0.3  # thetap
partisan_sd = 0.01

max_steps = 10000

ret = {}

def make_coin_list(bias, max_steps, num_coins=1):
    return [np.random.binomial(num_coins, bias) for _ in range(max_steps)]

coinslist = make_coin_list(bias=0.6, max_steps=10000)

def first_k_with_value_then_random(value, k):
    # eg: value = 5, k = 3 => "5,5,5,5,5,r" (the rest are treated as random, if any.)
    return ",".join([str(value)] * k) + ',r'

def flist_to_str(lst: list[float]):
    return ','.join([str(x) for x in lst])

def run(num_partisans):

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
        "true_bias": 0.6,
        "tosses_per_iteration": 1,
        "learning_rate": 0.25,
        "asymptotic_learning_max_iters": 99,
        "DWeps": 1,
        "disruption": num_partisans,
        "log": True,  # need to log to get mean_list
        "coinslist": coinslist,
        "break_on_asymptotic_learning": False,
    }


    res = run_ensemble(runs=runs, gen_graph=make_graph_generator(n, "allies"), sim_params=sim_params,
                        title=f"{frac_partisans:.2f} Partisans")
    
    # use mean[num_partisans:] for all non-partisans
    non_partisans_mean = [mean[num_partisans:].tolist() for mean in res[0].mean_list]
    asymptotic = res[0].asymptotic


    out = dict(partisans=round(frac_partisans, 2), asymptotic=asymptotic, non_partisans_mean=non_partisans_mean)

    logger.info(json.dumps(out))


def main():
    time1 = timeit.default_timer()

    with multiprocessing.Pool(10) as p:
        res = p.map(run, [x for x in num_partisans_sweep])
        print(res)
    
    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')

if __name__ == '__main__':
    main()