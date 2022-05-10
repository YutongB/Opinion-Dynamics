
import multiprocessing
from src.simulation.graphs import make_graph_generator
from src.simulation.runsim import run_ensemble


class color:
    # https://stackoverflow.com/questions/8924173/how-to-print-bold-text-in-python
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def printb(*arg1, **argv):
    print(color.BOLD + " ".join([str(x) for x in arg1]) + color.END, **argv)


n = 100
runs = 1000
asymp_max_iters = 99

command = f"python3 -m src.cli.simulate -n {n} -e allies -b 0.6 -r {runs} --nolog "
partisan_mean = 0.3
partisan_sd = 0.01

ret = {}

def first_k_with_value_then_random(value, k):
    # eg: value = 5, k = 3 => "5,5,5,5,5,r" (the rest are treated as random, if any.)
    return ",".join([str(value)] * k) + ',r'

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
        "max_steps": 1000,
        "true_bias": 0.6,
        "tosses_per_iteration": 1,
        "learning_rate": 0.25,
        "asymptotic_learning_max_iters": asymp_max_iters,
        "DWeps": 1,
        "disruption": num_partisans,
        "log": None
    }

    res = run_ensemble(runs=runs, gen_graph=make_graph_generator(n, "allies"), sim_params=sim_params,
                        title=f"{frac_partisans:.2f} Partisans")

    # this_command = command + f"--mean {means},r --sd {sds},r --disruption {num_partisans}"
    # printb(frac_partisans, this_command)

    # with open(os.devnull, 'wb') as devnull:
        # subprocess.check_call(this_command.split(), stdout=devnull, stderr=subprocess.STDOUT)

    ensemble_t_A = [sim.steps if sim.asymptotic[-1] == asymp_max_iters else -1 for sim in res]
    num_asymptotic = len(ensemble_t_A) - ensemble_t_A.count(-1)

    frac_asymptotic = num_asymptotic / len(ensemble_t_A)
    printb(f"partisans={frac_partisans:.2f} asymp={frac_asymptotic:.3f} runs={runs}")

    return (frac_partisans, frac_asymptotic)

def main():
    with multiprocessing.Pool(10) as p:
        res = p.map(run, range(1, n))
        print(res)

if __name__ == '__main__':
    main()