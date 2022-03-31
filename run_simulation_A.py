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
from dynamic import list_of_edges, modify_edge_weight
from simulation_numba import *
from simulation import *

if __name__ == '__main__':
#value settings:
    params = {'Number of runs': 3000, 'Number of iterations': 10000, 'Number of nodes': 5, 'Minimum dynamics':0.0,
                'Maximum dynamics': 0.2, 'Dynamics interval': 0.0005, 'Simple edge': True, 'Modification probability': True, "Include zero": True}
    num_runs = params['Number of runs']
    num_iter = params['Number of iterations']
    num_node = params['Number of nodes']
    min_dyn = params['Minimum dynamics']
    max_dyn = params['Maximum dynamics'] #exclusive
    dyn_interval = params['Dynamics interval']
    dyn_frac = np.arange(min_dyn, max_dyn, dyn_interval)
    simple_edge = params['Simple edge']
    mod_prob = params['Modification probability']
    include_zero = params['Include zero']

    output_dir = f'N{num_node}_test_results_finetune/'

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
    param_file.write(f"Modification probablity? {mod_prob}\n")
    param_file.write("(if True: dynamics = probability that a randomly chosen edge changes its value at each timestep i.e. A_ij does not necessarily change at each timestep)\n")
    param_file.write("(if False: dynamics = fraction of random selected edges in the network changes its value at each timestep) i.e. A_ij is guaranteed to change at each timestep\n")
    param_file.write(f"Include zero? (if False, network is a complete graph) {include_zero}\n")

    param_file.close()

    edge_list = list_of_edges(num_node)


    for frac in dyn_frac:
        runs = 0

        file = open(output_dir+f'F{frac:.4f}.dat', 'a') #File contains (in columns) for probability F: run number, asymptotic time, modification time step, state of N3 network after modification

        #edge_file = open(output_dir+f'edges_F{frac:.4f}.dat', 'a')

        print("Running simulation for dynamic probability {:.4f}".format(frac))
        asymp_time_list = []

        while runs < num_runs:

            g = nx.complete_graph(num_node)

            g = randomise_edge_weight(g, edge_weight=(-1, 1), simple_edge_weight=True, include_zero=include_zero)

            adj = get_adj_from_nx(g)

            beliefs = generate_beliefs(adj_size=num_node)

            asymp_t, _, _, _, modified_edges, modification_timestep = do_simulation_dynamic(adj, edge_list, beliefs, frac, true_bias=0.5, threshold=0.01, time_threshold=10, num_iter=num_iter, learning_rate=0.25, edge_weight=(-1, 1), simple_edge_weight=simple_edge, mod_probability=mod_prob, include_zero=include_zero)

            asymp_time_list.append(asymp_t)

            modified_edges = [list(edge) for edge in modified_edges]

            file.write(f"{runs}\t{asymp_t}\n")

            #edge_file.write(f"{runs}\t{modification_timestep}\t{list(modified_edges)}\n")

            runs += 1

        avg_asymp_time, med_asymp_time, stddev_asymp_time, tot_runs, runs_eliminated, percentage_eliminated = average_t_asymp(asymp_time_list, num_iter)

        overall_output.write(f"{frac:.4f}\t{tot_runs}\t{avg_asymp_time}\t{med_asymp_time}\t{runs_eliminated}\t{percentage_eliminated}\n")
