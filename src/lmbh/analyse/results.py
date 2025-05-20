from collections import namedtuple
import json
import os
import numpy as np

from ..simulation.graphs import create_model_graph

####### Dumping to json #######

class JSONEncoder(json.JSONEncoder):
    # from dumping a numpy array to json : https://stackoverflow.com/a/47626762
    def default(self, obj):
        if isinstance(obj, type(namedtuple)):
            return self.default(obj._asdict())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # list of ndarrays
        if isinstance(obj, list):
            return [self.default(el) for el in obj]
        return json.JSONEncoder.default(self, obj)


def dump_json(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        # serialise results to be dumped to JSON
        json.dump(results, f, cls=JSONEncoder)




def read_graph(adjacency, friendliness):
    g = create_model_graph()
    n = len(adjacency)
    g.add_vertex(n)
    g.add_edge_list([(u, v, friendliness[u][v])
                     for u, v in matrix_to_edge_list(adjacency)
                     if adjacency[u][v]],
                    eprops=[g.ep.friendliness])

    return g
    # friend_edges = {(u, v): friendliness[u][v]
    #    for u in range(n) for v in range(u+1, n) }


results_params = ("steps", "asymptotic", "agent_is_asymptotic", "coins", "mean_list", "std_list", "final_distr", "initial_distr", "friendliness", "adjacency", "distrs", "others")
# parameters that should be interpreted as np.array
results_array_params = ("agent_is_asymptotic", "adjacency", "friendliness", "final_distr", "initial_distr", "mean_list", "std_list", "distrs")

# from dataclasses import dataclass, field

# @dataclass
# class SimResults:
#     steps = 0
#     asymptotic = field(default_factory=list)
#     agent_is_asymptotic
#     coins
#     mean_list
#     std_list
#     final_distr

SimResults = namedtuple("SimulationResults", results_params, defaults=(None,))
SimulationResults = SimResults # with hout this line, we can't pickle SimResults. 

def parse_result(results):
    results_dict = {}
    for i, label in enumerate(results_params):
        results_dict[label] = results[i]

    for k in results_array_params:
        results_dict[k] = np.asarray(results_dict[k])

    return SimResults(**results_dict)


def read_results(filename):
    with open(filename, 'r') as f:
        top = json.load(f)
        # results, sim_params, args
        top['results'] = [parse_result(res) for res in top['results']]
        return top


def matrix_to_edge_list(mat):
    n = len(mat)
    return [(u, v)
            for u in range(n) for v in range(u+1, n)]

