import argparse

def parser_handle_graph(parser):
    parser.add_argument("-n", "--size", default=10, type=int, help="graph size")

    # TODO: overhaul how we read in edges vs friendliness
    #       currently -e corresponds to how we define friendliness
    #       we would like to have friends/enemies defined by ratio 
    #       (eg: 10^4 friends:1 enemy), so this option needs to allow both
    #       enum & number or something like that
    parser.add_argument('-e', "--edges", choices=["friends", "enemies", "random", "random_unif"], help="type of each edge (friendliness) on generated graph", default="random")

    # parser.add_argument("-l", "--leader", default="none", choices=["none", "for", "against"], help="generate a directed graph with a leader")

    parser.add_argument('-x', "--mode", choices=["complete", "balanced"], help="what way to run the simulations", default="complete")

    parser.add_argument('--test_DW', action="store_true", help="test various values of DWeps in same simulation")

    parser.add_argument("--DWeps", type=float, help="Threshold for ignoring the opinion for DW update rule; use 1 to use normal update rule", default=1)

    mean = parser.add_mutually_exclusive_group()
    mean.add_argument("--mean_range", nargs=2, default=[0.0, 1.0], type=float,
    metavar=("MEAN_FROM", "MEAN_TO"))
    mean.add_argument("--prior_mean", "--mean", type=float)

    sd = parser.add_mutually_exclusive_group()
    sd.add_argument("--sd_range", nargs=2, type=float, metavar=("SD_FROM", "SD_TO"))
    sd.add_argument("--prior_sd", "--sd", type=float, metavar="SD")
    sd.add_argument("--fwhm_range", nargs=2, default=[0.2, 0.8], type=float, metavar=("FWHM_FROM", "FWHM_TO"))
    sd.add_argument("--prior_fwhm", "--fwhm", type=float, metavar="FWHM")

def parser_handle_ensemble(parser):
    ensemble = parser.add_mutually_exclusive_group()
    ensemble.add_argument("--single", action="store_true", help="do a single simulation")
    ensemble.add_argument("-r", "--runs", default=20, type=int,
                        help="number of simulation runs in the ensemble")

def parse_args():
    parser = argparse.ArgumentParser(description='Opinion dynamics simulation, Yutong Bu 2021',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_handle_graph(parser)

    parser_handle_ensemble(parser)

    parser.add_argument("-m", "--max_iter", "--max_steps", default=1000, type=int,
                        help="maximum number of iterations per simulation")
    
    parser.add_argument("-b", "--bias", "--true_bias", default=0.5, type=float, help="true bias of coin")

    parser.add_argument("-a", "--asym-max-iters", "--asymptotic_max_iters", 
                        default=10, type=int, help="number of iterations for asymptotic learning", metavar="ASYM_MAX_ITERS")

    parser.add_argument("-t", "--tosses",  "--tosses-per-iter", 
                        default=10, type=int, help="number of tosses per iteration")

    # TODO: parameter for dice (n)

    parser.add_argument("-u", "--learning-rate", "--mu", default=0.5, type=float,
                        help="", metavar="MU")

    parser.add_argument("-f", "--fname", help="filename with results to read")

    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("number of ensemble simulation runs must be >=1")
    
    def prior_valid_length(lst, arg):
        if lst is not None and (len(lst) != args.size or len(lst) != 1):
            parser.error(f"{arg} must be given with either 1 or {args.size} values.  Consider using the range parameter.")

    prior_valid_length(args.prior_mean, "prior mean")
    prior_valid_length(args.prior_sd, "prior sd")

    return args