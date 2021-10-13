"""
Initialise simulation parameters
- Generate coin tosses
- Generate initial prior distributions

"""


import numpy as np
import argparse
from parseargs import parser_handle_ensemble, parser_handle_initial
from sim import dump_json, timestamp, generate_priors

"""


"""

def toss_coins_binom(bias=0.5, num_coins=1):
    return np.random.binomial(num_coins, bias)


def toss_coins_gen(toss_coins, *args, **kwargs):
    def t(*args, **kwargs):
        while True:
            yield toss_coins(*args, **kwargs)

    while True:
        yield t(*args, **kwargs)


def make_coin_lists(num_lists, max_iter, toss_coins, *args, **kwargs):
    # example call: make_coin_lists(10, 1000, toss_coins_binom, 0.5, 1)
    #  will make 10 coin lists each with 1000 coins tossed binomially with bias 0.5 and with 1 coin.
    return [[toss_coins(*args, **kwargs) for _ in range(max_iter)]
            for _ in range(num_lists)]


def main():

    parser = argparse.ArgumentParser(description='Opinion dynamics initial parameter generator, Yutong Bu 2021',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-n", "--size", default=10,
                        type=int, help="graph size")
    parser.add_argument("--mean_range", nargs=2,
                      default=[0.0, 1.0], type=float, metavar=("MEAN_FROM", "MEAN_TO"))
    parser.add_argument("--fwhm_range", nargs=2,
                    default=[0.2, 0.8], type=float, metavar=("FWHM_FROM", "FWHM_TO"))
                    
    parser_handle_ensemble(parser)
    parser_handle_initial(parser)

    args = parser.parse_args()

    runs = args.runs if not args.single else 1

    priors = [generate_priors(args.size, args.mean_range, args.fwhm_range) for run in range(runs)]

    lists = make_coin_lists(
        num_lists=runs,
        max_iter=args.max_iter,
        toss_coins=toss_coins_binom,
        bias=args.bias,
        num_coins=args.tosses,
    )
    fname = args.initfile or "output/initial-{}.json".format(timestamp())

    args = vars(args)
    del args["single"]
    del args["initfile"]
    args["runs"] = runs

    result = dict(coinlist=lists, priors=priors, args=args)

    dump_json(result, fname)

    print(f"Wrote initial parameters to {fname} with args", args)


if __name__ == '__main__':
    main()
