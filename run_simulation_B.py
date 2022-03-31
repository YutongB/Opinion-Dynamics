from numba import njit, prange
import numpy as np
import scipy as sp
import timeit
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from generate_graph import visualise_graph
import itertools
import json
from read_graph import read_coin_observations, recreate_graph
from scipy.optimize import curve_fit
from scipy import stats
from dynamic import *
from simulation_numba import *
from simulation import *

if __name__ == '__main__':
#value settings:
    params = {'Number of runs': 10000, 'Number of iterations': 10000, 'Number of nodes': 9, 'Minimum dynamics':0.0,
                'Maximum dynamics': 0.02, 'Dynamics interval': 0.0005, 'Simple edge': True, 'Dynamic type': 'P', "Include zero": False, "Barabasi_m": 3, "Barabasi_seed": 100}
    num_runs = params['Number of runs']
    num_iter = params['Number of iterations']
    num_node = params['Number of nodes']
    min_dyn = params['Minimum dynamics']
    max_dyn = params['Maximum dynamics'] #exclusive
    dyn_interval = params['Dynamics interval']
    dyn_prob = np.arange(min_dyn, max_dyn, dyn_interval)
    simple_edge = params['Simple edge']
    dyn_type = params['Dynamic type']
    include_zero = params['Include zero']
    barabasi_m = params['Barabasi_m']
    barabasi_seed = params['Barabasi_seed']

    #output_dir = f'N{num_node}_test_results_dynamicP/'
    output_dir = f'/Users/leeyishuen/Desktop/PhD/Opinion_Dynamics/dynamic_networks/Opinion-Dynamics/dynamicP_N9_test/N9_test_results_dynamicP_m3/'

    try:
        overall_output = open(output_dir+'overall_output.dat', 'a') #File contains (in columns) F, total number of runs, avg asymptotic time, median asymptotic time, number of eliminated runs, percentage eliminated
    except:
        os.mkdir(output_dir)
        overall_output = open(output_dir+'overall_output.dat', 'a') #File contains (in columns) F, total number of runs, avg asymptotic time, median asymptotic time, number of eliminated runs, percentage eliminated

    reproducable_param = open(output_dir+"params.json","w")
    json.dump(params, reproducable_param)
    reproducable_param.close()

    #Write description of parameters in a text file
    param_file = open(output_dir+'run_params.dat', 'w')

    param_file.write(f"Number of runs for each dynamic probability: {num_runs}\n")
    param_file.write(f"Number of iterations before termination: {num_iter}\n")
    param_file.write(f"Number of nodes: {num_node}\n")
    param_file.write(f"Minimum dynamic : {min_dyn}\n")
    param_file.write(f"Maximum dynamic : {max_dyn}\n")
    param_file.write(f"Dynamic interval : {dyn_interval}\n")
    param_file.write(f"Simple Edges? (i.e. edge take values of -1, 0 and 1 only) {simple_edge}\n")
    param_file.write(f"Dynamic type : {dyn_type}\n")
    param_file.write("(Type F dynamics = probability that a randomly chosen edge changes its value at each timestep i.e. A_ij does not necessarily change at each timestep)\n")
    param_file.write("(Type P dynamics = probability of each non-zero edge changing its value at each time step i.e. every single non-zero edge has a possibility to change in value at each timestep)\n")
    param_file.write(f"Include zero? (if False, network topology does not change i.e. zero edges stay zero etc.) {include_zero}\n")
    param_file.write(f"Barabasi_m (Number of edges to attach from a new node to existing nodes, if network is BA) {barabasi_m}\n")
    param_file.write(f"Barabasi_seed (Seed for RNG, same seed generates same barabasi networks with same topology) {barabasi_seed}\n")

    param_file.close()

    save_edges = open(output_dir+'save_edges.dat', 'w')

    edge_list = list_of_edges(num_node)

    initialise_g = nx.barabasi_albert_graph(num_node, barabasi_m, seed=barabasi_seed)

    g_nonzero_edges = [(i,j) for i, j in initialise_g.edges()]

    save_edges.write(f"{g_nonzero_edges}")

    save_edges.close()

    for prob in dyn_prob:
        runs = 0

        file = open(output_dir+f'F{prob:.4f}.dat', 'a') #File contains (in columns) for probability F: run number, asymptotic time, modification time step, state of N3 network after modification

        #file_ext = open(output_dir+f'F{prob:.4f}file_ext.dat', 'a') #Extended output, WARNING: Do not activate this line if running >100 runs per dyn_prob/large networks, it takes up a lot of space!

        print("Running simulation for dynamic probability {:.4f}".format(prob))
        asymp_time_list = []

        while runs < num_runs:

            g = initialise_g.copy()

            g = randomise_edge_weight(g, edge_weight=(-1, 1), simple_edge_weight=True, include_zero=include_zero)

            adj = get_adj_from_nx(g)

            beliefs = generate_beliefs(adj_size=num_node)

            asymp_t, _, _, _, modified_edge_count, modification_timestep = do_simulation_dynamic(adj, edge_list, beliefs, prob, dynamic_type='P', true_bias=0.6, threshold=0.01, time_threshold=100, num_iter=10000, learning_rate=0.25, edge_weight=(-1, 1), simple_edge_weight=True, include_zero=False)

            asymp_time_list.append(asymp_t)

            #modified_edges = [list(edge) for edge in modified_edges]

            file.write(f"{runs}\t{asymp_t}\n")
            #file_ext.write(f"{runs}\t{asymp_t}\t{modified_edge_count[:,0]}\t{modification_timestep}\n")

            #edge_file.write(f"{runs}\t{modification_timestep}\t{list(modified_edges)}\n")

            runs += 1

        avg_asymp_time, med_asymp_time, stddev_asymp_time, tot_runs, runs_eliminated, percentage_eliminated = average_t_asymp(asymp_time_list, num_iter)

        overall_output.write(f"{prob:.4f}\t{tot_runs}\t{avg_asymp_time}\t{med_asymp_time}\t{runs_eliminated}\t{percentage_eliminated}\n")
