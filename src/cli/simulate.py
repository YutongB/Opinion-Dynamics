import json
import timeit
from src.cli.parseargs import parse_args
from src.analyse.results import dump_json
from src.simulation.balance import run_balanced
from src.simulation.graphs import make_graph_generator
from src.simulation.runsim import run_ensemble, run_simulation
from src.utils import timestamp

def fwhm_to_sd(fwhm):
    # full width half mean = 2 * sqrt(2 * ln(2))
    if fwhm is None:
        return None
    return fwhm / 2.3548200450309493820231386529


def parse_prior(args):
    sd = fwhm_to_sd(args.prior_fwhm) if args.prior_sd is None else args.prior_sd
    sd_range = args.sd_range
    if sd_range is None:
        l, r = args.fwhm_range
        sd_range = fwhm_to_sd(l), fwhm_to_sd(r)
        assert sd_range[0] != None

    return {
        "mean": args.prior_mean,
        "sd": sd,
        "n": args.size,
        "mean_range": args.mean_range,
        "sd_range": sd_range,
    }


def parse_sim_params(args):
    return {
        "prior": parse_prior(args),
        "max_steps": args.max_iter,
        "true_bias": args.bias,
        "tosses_per_iteration": args.tosses,
        "learning_rate": args.learning_rate,
        "asymptotic_learning_max_iters": args.asym_max_iters,
        "DWeps": args.DWeps,
        "disruption": args.disruption,
        "log": None if args.mode == 'balanced' else True
    }

def main():
    time1 = timeit.default_timer()
   
    args = parse_args()

    print("Args parsed:", args)

    gen_graph = make_graph_generator(n=args.size, edges=args.edges)
    sim_params = parse_sim_params(args)

    print("Sim params:", sim_params)

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
            if args.initfile is not None:
                sim_params["coinslist"] = sim_params["coinslists"][0]
                del sim_params["coinslists"]
                sim_params["prior_mean"], sim_params["prior_sd"] = sim_params["priors"][0]
                del sim_params["priors"]

            res = [run_simulation(gen_graph(), **sim_params)]
        elif runs < 1:
            raise Exception("invalid number of runs")
        else:
            res = run_ensemble(runs=runs, gen_graph=gen_graph, sim_params=sim_params)
    elif args.mode == 'balanced':
        # res = run_balanced(sim_params=sim_params, threshold=runs, n=args.size, m=args.edges)
        res = run_balanced(sim_params=sim_params, threshold=runs, n=args.size)


    fname = "output/res-{}.json".format(timestamp())
    assert(res != None)

    out = dict(results=res, args=vars(args), sim_params=sim_params)
    dump_json(out, fname)
    print("Wrote latest results to", fname)
    with open('output/last_results', 'w') as f:
        f.write(fname)
    

    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')


if __name__ == '__main__':
    main()

