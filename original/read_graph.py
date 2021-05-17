import os
import networkx as nx
import numpy as np


# TODO might be much easier and faster to simply shelve the Simulation class?


def create_output_directory(parent_direc):
    direc_list = ['converge', 'data', 'mean', 'mode', 'network', 'prior', 'stddev', 'time_series']
    for direc in direc_list:
        os.makedirs(f'{parent_direc}/{direc}', exist_ok=True)


def remove_files(direc, starts_with='', ends_with=''):
    file_list = [f for f in os.listdir(direc) if (f.endswith(ends_with) and f.startswith(starts_with))]
    for f in file_list:
        os.remove(os.path.join(direc, f))


def extract_basic_info(direc):
    number_of_nodes = 0
    number_of_bias = 0
    time_list = []
    # figure out some preliminary parameters used for this graph
    with open(f'{direc}/data/graph.txt', 'r') as f:
        for i, line in enumerate(f):
            # the number of nodes is at the first line, last word
            if i == 0:
                number_of_nodes = int(line.split()[-1])
                break

    with open(f'{direc}/data/node0.txt', 'r') as f:
        lines = f.readlines()
        number_of_bias = len(lines[0].split())
        time_list = range(len(lines))

    bias_list = np.linspace(0, 1, number_of_bias)

    return number_of_nodes, time_list, bias_list


def extract_true_coin_bias(direc):
    with open(direc + '/data/observations.txt', 'r') as f:
        first_line = f.readline()
        true_bias = float(first_line.split()[-1])

    return true_bias


def recreate_graph(direc, prior_info=-2):
    number_of_nodes, time_list, bias_list = extract_basic_info(direc)
    g = nx.Graph()

    # add nodes first. Necessary to account for nodes that are isolated
    for n in range(number_of_nodes):
        g.add_node(n)

    with open(f'{direc}/data/graph.txt', 'r') as f:
        lines = f.readlines()
        del lines[0]  # first line has no edge data

    for line in lines:
        u, v, correlation = line.split()
        u = int(u)
        v = int(v)
        correlation = float(correlation)
        g.add_edge(u, v)
        g.edges[u, v]['correlation'] = correlation

    if prior_info >= -1:
        # extract prior from t=prior_info
        for n in range(number_of_nodes):
            g.nodes[n]['prior'] = extract_prior(direc, n, prior_info)

    return g


def extract_prior(direc, node=0, t=0):
    # NOTE: highly inefficient if the whole text file will eventually be used
    with open(f'{direc}/data/node{node}.txt', 'r') as f:
        lines = f.readlines()
        return np.array([float(data) for data in lines[t].split()])


def read_coin_observations(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        del lines[:2]

        heads_list = []
        tosses_list = []
        for line in lines:
            data = line.split()
            heads_list.append(int(data[1]))
            tosses_list.append(int(data[2]))

        return heads_list, tosses_list

