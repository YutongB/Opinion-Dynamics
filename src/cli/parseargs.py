import argparse


def parser_handle_graph(parser):
    parser.add_argument("-n", "--size", default=10,
                        type=int, help="graph size")

    parser.add_argument('-e', "--edges", choices=["friends", "allies", "enemies", "random", "random_unif"],
                        help="type of each edge (friendliness) on generated graph", default="random")

    # parser.add_argument("-l", "--leader", default="none", choices=["none", "for", "against"], help="generate a directed graph with a leader")

    parser.add_argument('-x', "--mode", choices=["complete", "balanced"],
                        help="what way to run the simulations", default="complete")

    parser.add_argument("--mean_range", nargs=2,
                      default=[0.0, 1.0], type=float, metavar=("MEAN_FROM", "MEAN_TO")
                      , help="generate initial prior distributions with means in this range")
    parser.add_argument("--prior_mean", "--mean", type=str, 
                        help="generate prior distrs with these means, if <n means specified, the rest have the same mean as the last one")

    sd_range = parser.add_mutually_exclusive_group()
    sd_range.add_argument("--sd_range", nargs=2, type=float,
                    metavar=("SD_FROM", "SD_TO")
                    , help="generate initial prior distributions with sds in this range")
    sd_range.add_argument("--fwhm_range", nargs=2,
                    default=[0.2, 0.8], type=float, metavar=("FWHM_FROM", "FWHM_TO"))
                    
    sd = parser.add_mutually_exclusive_group()
    sd.add_argument("--prior_sd", "--sd", type=str, metavar="SD",
                        help="generate prior distrs with these sds, if <n sds specified, the rest have the same sd as the last one")
    sd.add_argument("--prior_fwhm", "--fwhm", type=str, metavar="FWHM")


def parser_handle_ensemble(parser):
    ensemble = parser.add_mutually_exclusive_group()
    ensemble.add_argument("--single", action="store_true",
                          help="do a single simulation")
    ensemble.add_argument("-r", "--runs", default=20, type=int,
                          help="number of simulation runs in the ensemble")


def parser_handle_initial(parser):
    parser.add_argument("--initfile", help="filename with initial parameters")

    parser.add_argument("-m", "--max_iter", "--max_steps", default=1000, type=int,
                        help="maximum number of iterations per simulation")

    parser.add_argument("-t", "--tosses",  "--tosses-per-iter",
                        default=10, type=int, help="number of tosses per iteration")

    parser.add_argument("-b", "--bias", "--true_bias",
                        default=0.5, type=float, help="true bias of coin")


def parse_args():
    parser = argparse.ArgumentParser(description='Opinion dynamics simulation, Yutong Bu 2021',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_handle_graph(parser)

    parser_handle_ensemble(parser)

    parser_handle_initial(parser)

    # parser.add_argument('-s', "--seed", type=int, help="Random seed to seed the random number generators", default=None)

    parser.add_argument("--DWeps", type=float,
                        help="Threshold for ignoring the opinion for DW update rule; use 1 to use normal update rule", default=1)
    parser.add_argument("--disruption", type=int,
                        help="number of nodes keeping the same", default=0)
    parser.add_argument("-a", "--asym-max-iters", "--asymptotic_max_iters",
                        default=10, type=int, help="number of iterations for asymptotic learning", metavar="ASYM_MAX_ITERS")

    parser.add_argument("-u", "--learning-rate", "--mu", default=0.25, type=float,
                        help="", metavar="MU")

    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("number of ensemble simulation runs must be >=1")

    return args
