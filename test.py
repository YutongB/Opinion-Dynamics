import timeit
from sim import *
import argparse
from pprint import pformat

def parse_graph(args):
    n = args.size
    edges = args.edges
    if edges == "friends":
        generator = lambda: ADJ_FRIEND
    elif edges == "enemies":
        generator = lambda: ADJ_ENEMY
    elif edges == "random":
        generator = gen_relationship_binary
    elif edges == "random_unif":
        generator = gen_relationship_uniform

    return lambda: gen_complete_graph(n, generator)

def parse_sim_params(args):
    n = args.size

    prior_mean = args.prior_mean
    prior_sd = args.prior_sd
    
    if prior_mean is None:
        prior_mean = gen_prior_mean(n, mean_range=args.mean_range)

    if prior_sd is None:
        prior_sd = gen_prior_sd(n, fwhm_range=args.fwhm_range, sd_range=args.sd_range)

    return {
        "max_steps": args.max_iter,
        "true_bias": args.bias,
        "tosses_per_iteration": args.tosses,
        "learning_rate": args.learning_rate,
        "asymptotic_learning_max_iters": args.asym_max_iters,
        "prior_mean": prior_mean,
        "prior_sd": prior_sd,
    }

def main(args):

    time1 = timeit.default_timer()

    gen_graph = parse_graph(args)
    sim_params = parse_sim_params(args)
    runs = args.runs if not args.single else 1

    print("Starting {}\ngenerating complete graph of {} nodes with {} relationship\nsimulation parameters\n{}".format(
        f"ensemble of {runs} simulations" if not args.single else "simulation", 
        args.size, args.edges, pformat(sim_params)))

    if runs == 1:
        res = run_simulation(gen_graph(), **sim_params)
    elif runs < 1:
        raise Exception("invalid number of runs")
    else:
        res = do_ensemble(runs=runs, gen_graph=gen_graph, sim_params=sim_params)

    # TODO: Write res to file

    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Opinion dynamics simulation, Yutong Bu 2021')

    parser.add_argument("-m", "--max_iter", "--max_steps", default=1000, type=int,
                        help="maximum number of iterations per simulation")
    
    parser.add_argument('-e', "--edges", choices=["friends", "enemies", "random", "random_unif"], help="select edges type", default="random")

    # TODO: options for stuff other than complete graphs

    ensemble = parser.add_mutually_exclusive_group()
    ensemble.add_argument("--single", action="store_true", help="do a single simulation")
    ensemble.add_argument("-r", "--runs", default=20, type=int,
                        help="number of simulation runs in the ensemble")

    parser.add_argument("-n", "--size", default=10, type=int,
                        help="graph size")

    parser.add_argument("-b", "--bias", "--true_bias", default=0.5, type=float,
                        help="true bias of coin")

    parser.add_argument("-a", "--asym-max-iters", "--asymptotic_max_iters", 
                        default=10, type=int, help="true bias of coin")

    parser.add_argument("-t", "--tosses",  "--tosses-per-iter", 
                        default=10, type=int, help="number of tosses per iteration")

    # TODO: parameter for dice (n)

    parser.add_argument("-u", "--learning-rate", "--mu", default=0.5, type=float,
                        help="")

    mean = parser.add_mutually_exclusive_group()
    mean.add_argument("--mean_range", nargs=2, default=[0.0, 1.0], type=float)
    mean.add_argument("--prior_mean", "--mean", type=float)

    sd = parser.add_mutually_exclusive_group()
    sd.add_argument("--sd_range", nargs=2, type=float)
    sd.add_argument("--prior_sd", "--sd", type=float)
    sd.add_argument("--fwhm_range", nargs=2, default=[0.2, 0.8], type=float)
    sd.add_argument("--prior_fwhm", "--fwhm", type=float)

    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("number of ensemble simulation runs must be >=1")
    

    def prior_valid_length(lst, arg):
        if lst is not None and (len(lst) != args.size or len(lst) != 1):
            parser.error(f"{arg} must be given with either 1 or {args.size} values.  Consider using the range parameter.")

    prior_valid_length(args.prior_mean, "prior mean")
    prior_valid_length(args.sd_range, "sd mean")

    main(args)

