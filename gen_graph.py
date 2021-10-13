from sim import adjacency_mat, friendliness_mat, timestamp, dump_json, init_simulation
from simulate import parse_graph, parse_prior
from parseargs import parser_handle_graph, parser_handle_ensemble
import argparse
from pprint import pformat

def gen_result(gen_graph, prior):
    g = gen_graph()

    return {
        "adjacency": adjacency_mat(g),
        "friendliness": friendliness_mat(g),
        "initial_distr": init_simulation(g, **prior),
    }

def main(args):
    gen_graph = parse_graph(args)
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

