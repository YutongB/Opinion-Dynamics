
import multiprocessing
import timeit

from src.cli.make_sim_params import first_k_with_value_then_random
from src.analyse.analyse import frac_asymptotic_system, num_asymptotic_agents, num_asymptotic_agents_theta
from src.simulation.graphs import make_graph_generator
from src.simulation.runsim import run_ensemble

import logging

# use logger for thread-safe file writing (otherwise they might overwrite each other)
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

logpath = f"output/bigexperiment-1.log"
ch = logging.FileHandler(logpath)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)


n, runs = 100, 1000
# n, runs = 100, 10
asymp_max_iters = 99
num_partisans_sweep = range(1, n)
# num_partisans_sweep = [40]


bias = 0.6           # theta0
partisan_mean = 0.3  # thetap
partisan_sd = 0.01
command = f"python3 -m src.cli.simulate -n {n} -e allies -b {bias} -r {runs} --nolog "

max_steps = 1000

ret = {}


def run(num_partisans):

    frac_partisans = num_partisans / n
    means = first_k_with_value_then_random(partisan_mean, num_partisans)
    sds = first_k_with_value_then_random(partisan_sd, num_partisans)

    sim_params = {
        "prior": {
            "mean": means,
            "sd": sds,
            "n": n,
            "mean_range": (0, 1),
            "sd_range": (0.2, 0.8),
        },
        "max_steps": max_steps,
        "true_bias": bias,
        "tosses_per_iteration": 1,
        "learning_rate": 0.25,
        "asymptotic_learning_max_iters": asymp_max_iters,
        "DWeps": 1,
        "disruption": num_partisans,
        "log": None,  # need to log to get mean_list
    }

    res = run_ensemble(runs=runs, gen_graph=make_graph_generator(n, "allies"), sim_params=sim_params,
                        title=f"{frac_partisans:.2f} Partisans")

    # frac_asymptotic = frac_asymptotic_system(res, asymp_max_iters)
    x = num_asymptotic_agents(res)
    close_agents = num_asymptotic_agents_theta(res, bias, partisan_mean)

    logger.info(f"partisans={frac_partisans:.2f} num_asymp_agents={x} close_agents={close_agents} runs={runs} n={n}")

    return (frac_partisans, x, close_agents)

def main():
    time1 = timeit.default_timer()

    with multiprocessing.Pool(10) as p:
        res = p.map(run, [x for x in num_partisans_sweep])
        print(res)
    
    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')

if __name__ == '__main__':
    main()