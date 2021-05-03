import timeit
from sim import *


def main():
    time1 = timeit.default_timer()

    res = do_ensemble(runs=20, gen_graph=lambda: complete_graph_of_random(15), sim_params={"max_steps": 10000})

    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')


if __name__ == '__main__':
    main()

