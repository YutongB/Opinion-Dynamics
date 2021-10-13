import timeit
from sim import *
from pprint import pformat
from parseargs import parse_args
from balance import *


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

def parse_prior(args):
    n = args.size

    prior_mean = args.prior_mean
    prior_sd = args.prior_sd
    
    if prior_mean is None:
        prior_mean = gen_prior_mean(n, mean_range=args.mean_range)
    elif len(prior_mean) == 1:  # set all nodes to have same prior_mean
        prior_mean = [prior_mean] * n

    if prior_sd is None:
        prior_sd = gen_prior_sd(n, fwhm_range=args.fwhm_range, sd_range=args.sd_range)
    elif len(prior_sd) == 1:  # set all nodes to have same prior_sd
        prior_sd = [prior_sd] * n

    return {
        "prior_mean": prior_mean,
        "prior_sd": prior_sd,
    }

def parse_sim_params(args):
    

    return {
        **parse_prior(args),
        "max_steps": args.max_iter,
        "true_bias": args.bias,
        "tosses_per_iteration": args.tosses,
        "learning_rate": args.learning_rate,
        "asymptotic_learning_max_iters": args.asym_max_iters,
        "DWeps": args.DWeps,
        "log": None if args.mode == 'balanced' else True
    }

def main():
    time1 = timeit.default_timer()
   
    args = parse_args()

    print(args)

    gen_graph = parse_graph(args)
    sim_params = parse_sim_params(args)

    runs = args.runs if not args.single else 1

    if args.initfile is not None:
        print("Using initial parameters f{args.initfile}.")
        with open(args.initfile, 'r') as f:
            initdata = json.load(f)
            coinslist = initdata["coinlist"]
            initargs = initdata["args"]
            priors = initdata["priors"]

            runs = initargs['runs']
            sim_params = {
                **sim_params,
                "max_steps": initargs["max_iter"],
                "true_bias": initargs["bias"],
                "tosses_per_iteration": initargs["tosses"],
                "coinslists": coinslist,
                "priors": priors
            }

    res = None

    if args.mode == "complete":
        print("Starting {}\ngenerating complete graph of {} nodes with {} relationship\n".format(
            f"ensemble of {runs} simulations" if not args.single else "simulation", 
            args.size, args.edges))

        if runs == 1:
            sim_params["coins"] = sim_params["coinslists"][0]
            sim_params["prior"] = sim_params["priors"][0]
            del sim_params["coinslists"]
            del sim_params["priors"]

            res = [run_simulation(gen_graph(), **sim_params)]
        elif runs < 1:
            raise Exception("invalid number of runs")
        else:
            res = do_ensemble(runs=runs, gen_graph=gen_graph, sim_params=sim_params)
    elif args.mode == 'balanced':
        # res = run_balanced(sim_params=sim_params, threshold=runs, n=args.size, m=args.edges)
        res = run_balanced(sim_params=sim_params, threshold=runs, n=args.size)


    fname = "output/res-{}.json".format(timestamp())
    assert(res != None)

    out = dict(res=res, args= vars(args), sim_params=sim_params)
    dump_json(out, fname)
    print("Wrote results to", fname)

    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')


if __name__ == '__main__':
    main()

