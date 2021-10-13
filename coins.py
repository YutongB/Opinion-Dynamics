import numpy as np
import argparse
from parseargs import parser_handle_ensemble, parser_handle_coins
from sim import dump_json, timestamp


def toss_coins_binom(bias=0.5, num_coins=1):
    return np.random.binomial(num_coins, bias)


def toss_coins_list(list):
    for toss in list:
        yield toss

def blah(lists):
    for list in lists:
        yield toss_coins_list(list)




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

    parser = argparse.ArgumentParser(description='Opinion dynamics coin generator, Yutong Bu 2021',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_handle_ensemble(parser)
    parser_handle_coins(parser)

    args = parser.parse_args()

    fname = args.coinfile or "output/coins-{}.json".format(timestamp())

    lists = make_coin_lists(
        num_lists=args.runs,
        max_iter=args.max_iter,
        toss_coins=toss_coins_binom,
        bias=args.bias,
        num_coins=args.tosses,
    )

    dump_json(lists, fname)


if __name__ == '__main__':
    main()
