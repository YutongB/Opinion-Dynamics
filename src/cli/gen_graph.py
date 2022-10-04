import argparse
from pprint import pformat

from src.cli.simulate import parse_prior
from src.cli.parseargs import parser_handle_graph, parser_handle_ensemble
from src.analyse.results import dump_json
from src.simulation.graphs import gen_complete_graph, get_edge_generator
from src.simulation.initsim import init_simulation
from src.simulation.sim import adjacency_mat, friendliness_mat
from src.utils import timestamp

def gen_result(gen_graph, prior):
    g = gen_graph()

    return {
        "adjacency": adjacency_mat(g),
        "friendliness": friendliness_mat(g),
        "initial_distr": init_simulation(g, **prior),
    }

def main(args):
    gen_graph = lambda: gen_complete_graph(args.size, get_edge_generator(args.edges))
    prior = parse_prior(args)

    print("Generating {} complete graph(s) of {} nodes with {} relationship\nprior\n{}".format(args.runs, args.size, args.edges, pformat(prior)))

    results = [gen_result(gen_graph, prior) for _ in range(args.runs)]
    

    fname = "output/graph-{}.json".format(timestamp())
    dump_json(results, fname)
    print("Wrote results to", fname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Opinion dynamics simulation graph generator, Yutong Bu 2021')
    
    parser_handle_graph(parser)
    parser_handle_ensemble(parser)

    args = parser.parse_args()

    def prior_valid_length(lst, arg):
        if lst is not None and (len(lst) != args.size or len(lst) != 1):
            parser.error(f"{arg} must be given with either 1 or {args.size} values.  Consider using the range parameter.")

    prior_valid_length(args.prior_mean, "prior mean")
    prior_valid_length(args.sd_range, "sd mean")

    main(args)

