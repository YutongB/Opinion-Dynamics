import json
from typing import List
from src.simulation.initsim import gen_prior_param
from src.utils import timestamp
import numpy as np

def make_coin_list(bias, max_steps, num_coins=1):
    return [np.random.binomial(num_coins, bias) for _ in range(max_steps)]
    # return [1] * 6000 + [0] * 4000

def first_k_with_value_then_random(value, k):
    # eg: value = 5, k = 3 => "5,5,5,5,5,r" (the rest are treated as random, if any.)
    return ",".join([str(value)] * k) + ',r'

def flist_to_str(lst: List[float]):
    return ','.join([str(x) for x in lst])

def get_sim_params():
    num_partisans = 5
    n = 10
    coinslist = None
    # coinslist = make_coin_list(bias=0.6, max_steps=1000)
    # coinslist = [1] * 6000 + [0] * 4000

    prior_mean = first_k_with_value_then_random(0.3, num_partisans)
    # prior_mean = flist_to_str(gen_prior_param(prior_mean, n, range=(0,1)))
    # prior_mean = '0.3'

    prior_sd = first_k_with_value_then_random(0.01, num_partisans)
    # prior_sd = flist_to_str(gen_prior_param(prior_sd, n, range=(0.2, 0.8)))
    # prior_sd = '0.5'

    sim_params = {
        "prior": {
            "mean": prior_mean,
            "sd": prior_sd,
            "n": n,
            "mean_range": (0,1),
            "sd_range": (0.2, 0.8),
        },
        "max_steps": 10000,
        "true_bias": 0.6,
        "tosses_per_iteration": 1,
        "learning_rate": 0.25,
        "asymptotic_learning_max_iters": 99,
        "DWeps": 1,
        "disruption": num_partisans,
        "log": True,  # need to log to get mean_list
        "coinslist": coinslist,
        "break_on_asymptotic_learning": True,
    }

    return sim_params

    
def main():
    sim_params = get_sim_params()
    with open(f"params/sim_params-{timestamp()}.json", 'w') as f:
        json.dump(sim_params, f)

