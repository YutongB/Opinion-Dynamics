import sys
sys.path.append("..") # Adds higher directory to python modules path.
from parseargs import parse_args
import networkx as nx
import numpy as np
import scipy.stats
from read_graph import remove_files, recreate_graph, create_output_directory, read_coin_observations
import timeit
from generate_graph import two_opposing_groups, unbalanced_triangle, visualise_graph
import matplotlib.pyplot as plt
import random
import math
import json

# TODO test how much time def write_prior actually takes


class Simulation:
    def __init__(self, graph, true_bias=0.5, tosses_per_iteration=10, output_direc='output/default', coin_obs_file=None, bias_len=21, coins=None):
        # TODO remove bias_len and infer it from the length of the prior
        self.g_init = nx.Graph(graph)
        # self.graph_evolution = graph_evolution
        self.true_bias = true_bias
        self.tosses_per_iteration = tosses_per_iteration
        self.output_direc = output_direc
        self.bias_list = np.linspace(0, 1, bias_len)

        if coins is not None:
            self.heads_list = coins
            self.tosses_list = [tosses_per_iteration] * len(coins)
        elif coin_obs_file is None:
            self.heads_list = []
            self.tosses_list = []
        elif coin_obs_file is True:
            self.heads_list, self.tosses_list = read_coin_observations('output/default/data/observations.txt')
        else:
            self.heads_list, self.tosses_list = read_coin_observations(coin_obs_file)

        create_output_directory(output_direc)


    def simulate_coin_toss(self, tosses=0, bias=0.5):
        if tosses > 0:
            return np.random.binomial(tosses, bias)
        else:
            return np.random.binomial(self.tosses_per_iteration, self.true_bias)


    def calculate_coin_probability(self, num_heads, num_tosses):
        return scipy.stats.binom.pmf(num_heads, num_tosses, self.bias_list)


    def observe_coin(self, same_obs=True):
        # observe coin once and update the priors
        if same_obs:
            num_heads = self.simulate_coin_toss()
            # num_heads = int(self.true_bias * self.tosses_per_iteration)
            prob = self.calculate_coin_probability(num_heads, self.tosses_per_iteration)
            for node in self.g:
                norm = np.sum(self.g.nodes[node]['prior'] * prob)
                self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob / norm

        else:
            raise NotImplementedError('Code for dealing with each node observing different tosses not implemented yet')

        return num_heads, self.tosses_per_iteration


    def blend_pos(self, learning_rate=0.25, method=0):
        if method == 0:  # weighted average
            # TODO merge learning_rate into edge_weight?
            for i in self.g:  # loop through nodes
                self_pos = self.g.nodes[i]['self_pos'].copy()
                new_pos = np.array(self_pos) * 0
                total_edge_weight = 0

                for j in self.g.neighbors(i):
                    if i == j:
                        continue
                    edge_weight = self.g.edges[i, j]['correlation']
                    neighbour_pos = np.array(self.g.nodes[j]['self_pos'])

                    diff = neighbour_pos - self_pos
                    new_pos += edge_weight * diff
                    total_edge_weight += np.abs(edge_weight)

                if total_edge_weight!=0:
                    if isinstance(learning_rate, (list, tuple, np.ndarray)):
                        new_pos *= learning_rate[i] / total_edge_weight
                    else:
                        new_pos *= learning_rate / total_edge_weight
                else: #if total edge weight is zero, we make no modification to the prior in the last time step
                    new_pos *= 0
                self_pos = np.maximum(1e-10, self_pos + new_pos)
                self_pos /= self_pos.sum()
                self.g.nodes[i]['prior'] = self_pos

        elif method == 1:  # bayesian-like update rule
            # TODO use edge weight instead of learning_rate
            for i in self.g:
                self_pos = self.g.nodes[i]['self_pos'].copy()
                for j in self.g.neighbors(i):
                    base_pos = np.ones(len(self.bias_list))
                    base_pos /= base_pos.sum()
                    edge_weight = self.g.edges[i, j]['correlation']
                    if edge_weight < 0:
                        neighbour_pos = (1 - self.g.nodes[j]['self_pos']) + base_pos
                    else:
                        neighbour_pos = self.g.nodes[j]['self_pos'] + base_pos

                    neighbour_pos /= neighbour_pos.sum()

                    self_pos *= neighbour_pos ** np.abs(learning_rate)
                    self_pos /= self_pos.sum()

                self.g.nodes[i]['prior'] = self_pos


    def write_prior(self):
        # save the new prior
        for i in self.g:
            with open(f'{self.output_direc}/data/node{i}.txt', 'a') as f:
                for data in self.g.nodes[i]['prior']:
                    f.write(str(data) + ' ')
                f.write('\n')

    def save_edges(self):
        # save the edges in each time step
        with open(f'{self.output_direc}/data/edges.txt', 'a') as f:
            for u, v, correlation in self.g.edges.data('correlation'):
                f.write('{} {} {}\n'.format(u, v, correlation))

    def save_graph(self):
        # save the priors into a new text file
        remove_files(f'{self.output_direc}/data', starts_with='node', ends_with='.txt')
        for node, prior in self.g.nodes(data='prior'):
            with open(f'{self.output_direc}/data/node{node}.txt', 'w+') as f:
                for i in prior:
                    f.write(str(i) + ' ')
                f.write('\n')

        # save information about the initial graph
        with open(f'{self.output_direc}/data/graph.txt', 'w+') as f:
            f.write('Number of nodes: {}\n'.format(self.g.number_of_nodes()))
            for u, v, correlation in self.g.edges.data('correlation'):
                f.write('{} {} {}\n'.format(u, v, correlation))


    def save_tosses(self):
        with open(f'{self.output_direc}/data/observations.txt', 'w+') as f:
            f.write('True coin bias: {}\n'.format(self.true_bias))
            f.write('Iteration Num_heads Num_tosses\n')
            for i in range(len(self.heads_list)):
                f.write(f'{i+1} {self.heads_list[i]} {self.tosses_list[i]}\n')


    def number_reached_asymptotic_learning(self, one_or_all_nodes='all'):
        """

        """
        threshold = 0.99
        reached = 0
        for node in self.g:
            if np.any(self.g.nodes[node]['prior'] >= threshold):
                reached += 1

        return reached


    def modify_edge_weight(self, f, edge_weight=(-1, 1), simple_edge_weight=True):
        #f is the fraction of edges (i.e. A_{ij}) that will be modified
        list_of_edges = [i for i in list(self.g.edges())]
        num_to_modify = int(f*len(list_of_edges))

        edges_to_modify = [i for i in random.choices(list_of_edges, k=num_to_modify)]
        #print(f, len(list_of_edges), num_to_modify, edges_to_modify)
        if simple_edge_weight:
            for edge in edges_to_modify:
                roll = random.choice([-1,0,1])

                if edge[0] == edge[1]:  # self weight cannot be negative
                    if roll < 0:
                        roll *= -1
                self.g.edges[edge[0], edge[1]]['correlation'] = roll
        else:
            edge_weight = put_smaller_number_first(edge_weight)
            if edges_to_modify:
                for edge in edges_to_modify:
                    roll = resized_range(edge_weight)
                    if edge[0] == edge[1]:  # self weight cannot be negative
                        if roll < 0:
                            roll *= -1
                    self.g.edges[edge[0], edge[1]]['correlation'] = roll


        return self.g

    def do_simulation(self, f=0.1, num_iter=10, learning_rate=0.25, blend_method=0, save_info=True, dynamic=True, simple_edge_weight=True):
        self.g = self.g_init.copy()
        #print(self.g.edges[0,1]['correlation'])
        #self.save_g = []
        #self.save_g.append(self.g_init)
        if save_info:
            self.save_graph()
            self.save_edges()

        t = 1
        asymptotic_learning_time = 0
        asymptotic_learning_time_threshold = 10

        if len(self.heads_list) > 0:
            # heads/tosses already determined, so follow that
            print('Using predetermined coin tosses')
            while t <= len(self.heads_list):
            # for t in range(len(self.heads_list)):

                if t % 1000 == 0:
                    print('Iteration {}'.format(t))
                prob = self.calculate_coin_probability(self.heads_list[t-1], self.tosses_list[t-1])
                for node in self.g:
                    norm = np.sum(self.g.nodes[node]['prior'] * prob)
                    self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob / norm

                self.blend_pos(learning_rate=learning_rate, method=blend_method)
                if save_info:
                    self.write_prior()

                if self.number_reached_asymptotic_learning() == self.g.number_of_nodes():
                    asymptotic_learning_time += 1
                    if asymptotic_learning_time >= asymptotic_learning_time_threshold:
                        #print(f'Reached asymptotic learning at iteration {t}.')
                        self.heads_list = self.heads_list[:t]
                        self.tosses_list = self.tosses_list[:t]
                        if save_info:
                            self.save_tosses()
                        return t
                else:
                    asymptotic_learning_time = 0

                if dynamic:
                    self.g = self.modify_edge_weight(f=f, edge_weight=(-1, 1), simple_edge_weight=simple_edge_weight)

                if save_info:
                    self.save_edges()
                #self.save_g.append(self.g)

                t += 1
            if t > len(self.heads_list) and t <= num_iter:
                print('All predetermined coin tosses used. Will now generate new tosses', len(self.heads_list))

        while t <= num_iter:
            # simulate our own coin tosses.
            #for t in range(num_iter):
            if t % 1000 == 0:
                print('Iteration {}'.format(t))
            num_heads, num_tosses = self.observe_coin()
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            self.heads_list.append(num_heads)
            self.tosses_list.append(num_tosses)
            if save_info:
                self.write_prior()
            if self.number_reached_asymptotic_learning() == self.g.number_of_nodes():

                asymptotic_learning_time += 1
                if asymptotic_learning_time >= asymptotic_learning_time_threshold:
                    # print(f'Reached asymptotic learning at iteration {t}.')
                    return t
                    break
            else:
                asymptotic_learning_time = 0

            if dynamic:
                self.g = self.modify_edge_weight(f=f, edge_weight=(-1, 1), simple_edge_weight=simple_edge_weight)

            if save_info:
                self.save_edges()

            t += 1

        if save_info:
            self.save_tosses()


    def do_simulation_without_signal(self, num_iter=10, learning_rate=0.25, blend_method=0):
        self.heads_list = []
        self.tosses_list = []
        self.save_graph()
        for t in range(num_iter):
            if t % 100 == 0:
                print('Iteration {}'.format(t))
            for i in self.g:
                self.g.nodes[i]['self_pos'] = self.g.nodes[i]['prior'].copy()

            self.heads_list.append(0)
            self.tosses_list.append(0)
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            self.write_prior()
            if self.number_reached_asymptotic_learning() == self.g.number_of_nodes():
                # print(f'Reached asymptotic learning at iteration {t}.')
                break

        self.save_tosses()

    def do_simulation_without_print_and_save(self, num_iter=1000, learning_rate=0.25, blend_method=0):
        # return True is asymptotic learning is achieved, False otherwise
        t = 1
        asymptotic_learning_time = 0
        asymptotic_learning_time_threshold = 10

        if len(self.heads_list) > 0:
            # heads/tosses already determined, so follow that
            while t <= len(self.heads_list):
                prob = self.calculate_coin_probability(self.heads_list[t-1], self.tosses_list[t-1])
                for node in self.g:
                    norm = np.sum(self.g.nodes[node]['prior'] * prob)
                    self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob / norm

                self.blend_pos(learning_rate=learning_rate, method=blend_method)
                if self.number_reached_asymptotic_learning() == self.g.number_of_nodes():
                    asymptotic_learning_time += 1
                    if asymptotic_learning_time >= asymptotic_learning_time_threshold:
                        return t
                else:
                    asymptotic_learning_time = 0
                t += 1

        while t <= num_iter:
            # simulate our own coin tosses.
            num_heads, num_tosses = self.observe_coin()
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            if self.number_reached_asymptotic_learning() == self.g.number_of_nodes():
                asymptotic_learning_time += 1
                if asymptotic_learning_time >= asymptotic_learning_time_threshold:
                    return t
            else:
                asymptotic_learning_time = 0
            t += 1

        return 0

    def do_simulation_without_signal_and_saves(self, num_iter=10, learning_rate=0.25, blend_method=0):
        for t in range(num_iter):
            for i in self.g:
                self.g.nodes[i]['self_pos'] = self.g.nodes[i]['prior'].copy()
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            if self.number_reached_asymptotic_learning() == self.g.number_of_nodes():
                # print(f'Reached asymptotic learning at iteration {t}.')
                break

    def do_simulation_and_check_individual_asymp(self, num_iter=10, learning_rate=0.25, blend_method=0, save_info=False):
        if save_info:
            self.save_graph()

        t = 1
        asymptotic_learning_time = 0
        asymptotic_learning_time_threshold = 10

        current_asymp_number = 0

        if len(self.heads_list) > 0:
            # heads/tosses already determined, so follow that
            print('Using predetermined coin tosses')
            while t <= len(self.heads_list):
            # for t in range(len(self.heads_list)):
                if t % 100 == 0:
                    print('Iteration {}'.format(t))
                prob = self.calculate_coin_probability(self.heads_list[t-1], self.tosses_list[t-1])
                for node in self.g:
                    norm = np.sum(self.g.nodes[node]['prior'] * prob)
                    self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob / norm

                self.blend_pos(learning_rate=learning_rate, method=blend_method)
                if save_info:
                    self.write_prior()

                number_asymp = self.number_reached_asymptotic_learning()
                if number_asymp > current_asymp_number:
                    print(f'{number_asymp} nodes achieved asymptotic learning at t={t}')
                    current_asymp_number = number_asymp

                if number_asymp == self.g.number_of_nodes():
                    asymptotic_learning_time += 1
                    if asymptotic_learning_time >= asymptotic_learning_time_threshold:
                        # print(f'Reached asymptotic learning at iteration {t}.')
                        self.heads_list = self.heads_list[:t]
                        self.tosses_list = self.tosses_list[:t]
                        if save_info:
                            self.save_tosses()
                        return
                else:
                    asymptotic_learning_time = 0
                t += 1
            if t > len(self.heads_list) and t <= num_iter:
                print('All predetermined coin tosses used. Will now generate new tosses', len(self.heads_list))

        while t <= num_iter:
            # simulate our own coin tosses.
            #for t in range(num_iter):
            if t % 100 == 0:
                print('Iteration {}'.format(t))
            num_heads, num_tosses = self.observe_coin()
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            self.heads_list.append(num_heads)
            self.tosses_list.append(num_tosses)
            if save_info:
                self.write_prior()

            number_asymp = self.number_reached_asymptotic_learning()
            if number_asymp > current_asymp_number:
                # print(f'{number_asymp} nodes achieved asymptotic learning at t={t}')
                current_asymp_number = number_asymp

            if number_asymp == self.g.number_of_nodes():
                asymptotic_learning_time += 1
                if asymptotic_learning_time >= asymptotic_learning_time_threshold:
                    #print(f'Reached asymptotic learning at iteration {t}.')
                    break
            else:
                asymptotic_learning_time = 0
            t += 1

        if save_info:
            self.save_tosses()

    def find_width(self):
        # find out how far the distribution of belief tilts away from the true bias. Also finds out if there is imbalance between the widths
        max_upper_width = 0.0
        max_lower_width = 0.0

        for n in self.g:
            prior = self.g.nodes[n]['prior']
            max_arg = np.argmax(prior)
            bias = max_arg * self.bias_list[1]
            width = self.true_bias - bias

            if width > max_upper_width:
                max_upper_width = width
            elif width < max_lower_width:
                max_lower_width = width

        return max_upper_width, max_lower_width


def put_smaller_number_first(number):
    return min(number), max(number)


def resized_range(number):
    return np.random.random() * (number[1] - number[0]) + number[0]
    #uniform sampling within range (number[0],number[1])


def gaussian(bias_list, mean=0.5, fwhm=0.5):
    not_norm = np.exp(-4 * np.log(2) * ((bias_list - mean) / fwhm) ** 2)
    return not_norm / not_norm.sum()


def make_evenly_spaced_priors(g, fwhm=0.4, bias_len=21):
    bias_list = np.linspace(0, 1, bias_len)
    mean_list = np.linspace(0, 1, g.number_of_nodes())

    for count, mu in enumerate(mean_list):
        g.nodes[count]['prior'] = gaussian(bias_list, mean=mu, fwhm=fwhm)

    return g

def set_prior_from_nparray(g, mat):
    for i in range(g.number_of_nodes()):
        g.nodes[i]['prior'] = mat[i]
    return g

def set_friendliness_from_nparray(g, mat):
    # set up friendliness from matrix
    for u in range(len(mat)):
        for v in range(u+1, len(mat)):
            g.edges[u, v]['correlation'] = mat[u][v]
    return g

def randomise_prior(g, mean=(0.0, 1.0), fwhm=(0.2, 0.8), bias_len=21):
    g = nx.Graph(g)
    bias_list = np.linspace(0, 1, bias_len)
    mean = put_smaller_number_first(mean)
    fwhm = put_smaller_number_first(fwhm)

    for i in range(g.number_of_nodes()):
        # randomise initial prior of each node. assume normal distribution.
        mean1 = resized_range(mean)
        fwhm1 = resized_range(fwhm)
        prior = np.exp(-4 * np.log(2) * ((bias_list - mean1) / fwhm1) ** 2)
        prior /= prior.sum()
        g.nodes[i]['prior'] = prior

    # if g.number_of_nodes() == 2:
    #
    #     mean1 = resized_range((mean[0],0.45))
    #     fwhm1 = resized_range((0.2,0.4))
    #     prior = np.exp(-4 * np.log(2) * ((bias_list - mean1) / fwhm1) ** 2)
    #     prior /= prior.sum()
    #     g.nodes[0]['prior'] = prior
    #
    #     mean1 = resized_range((0.55, mean[1]))
    #     fwhm1 = resized_range((0.2,0.4))
    #     prior = np.exp(-4 * np.log(2) * ((bias_list - mean1) / fwhm1) ** 2)
    #     prior /= prior.sum()
    #     g.nodes[1]['prior'] = prior


    return g


def initialise_edge_weight(g, edge_weight=(-1, 1)):
    #A_{ij} can now take any value between -1,1
    edge_weight = put_smaller_number_first(edge_weight)
    for i, j in g.edges():
        roll = resized_range(edge_weight)
        if i == j:  # self weight cannot be negative
             if roll < 0:
                 # NOTE this correction will not reflect input probability distribution if not sym around 0
                 roll *= -1
        g.edges[i, j]['correlation'] = roll

    return g

def randomise_edge_weight(g, edge_weight=(-1, 1), simple_edge_weight=True, include_zero=False):
    if simple_edge_weight and include_zero:
        for i, j in g.edges():
            # roll = random.choice([-1,0,1])
            roll = random.choice([-1,1])

            if i == j:  # self weight cannot be negative
                if roll < 0:
                    # NOTE this correction will not reflect input probability distribution if not sym around 0
                    roll *= -1

            g.edges[i, j]['correlation'] = roll

    else:
        edge_weight = put_smaller_number_first(edge_weight)
        for i, j in g.edges():
            roll = resized_range(edge_weight)
            if simple_edge_weight:
                if roll < 0:
                    roll = -1
                elif roll > 0:
                    roll = 1

            if i == j:  # self weight cannot be negative
                if roll < 0:
                    # NOTE this correction will not reflect input probability distribution if not sym around 0
                    roll *= -1
            g.edges[i, j]['correlation'] = roll

    return g

# def modify_edge_weight(g, f, edge_weight=(-1, 1)):
#     #f is the fraction of edges (i.e. A_{ij}) that will be modified
#     num_of_nodes = nx.number_of_nodes(g)
#     num_to_modify = int(f*num_of_nodes)
#     list_of_edges = list(g.edges())
#     edges_to_modify = random.choices(list_of_edges, k=num_to_modify)
#
#     edge_weight = put_smaller_number_first(edge_weight)
#     for edge in edges_to_modify:
#         roll = resized_range(edge_weight)
#         if edge[0] == edge[1]:  # self weight cannot be negative
#             if roll < 0:
#                 roll *= -1
#         g.edges[edge[0], edge[1]]['correlation'] = roll
#
#     return g


def generate_graph(number_of_nodes=10):
    """
    This generates a brand new graph. To reuse a similar graph, look at recreate_graph
    """
    # self.g = nx.gnp_random_graph(n=self.number_of_nodes, p=0.2)
    g = nx.complete_graph(number_of_nodes)

    # add self loops to represent self weight. Also make sure the graph is connected
    for node in g:
        if len(list(g.neighbors(node))) == 0:
            while number_of_nodes > 1:
                random_neighbor = np.random.randint(number_of_nodes)
                if random_neighbor != node:
                    g.add_edge(node, random_neighbor)
                    break

        g.add_edge(node, node)

    return g


def add_self_loops(g, add_weights=True):
    for node in g:
        g.add_edge(node, node)
        if add_weights:
            g.edges[node, node]['correlation'] = 1

    return g

# def graph_evolution(g, num_iter, f):
#     graph = g.copy()
#     graph_evolution = []
#     graph_evolution.append(g)
#     for i in range(0,num_iter):
#         graph = modify_edge_weight(graph, f, edge_weight=(-1, 1))
#         graph_evolution.append(graph)
#
#     return graph_evolution

def main(args):
    print(args)
    time1 = timeit.default_timer()
    # g = unbalanced_triangle()
    
    num_runs = args.runs if not args.single else 1  #20
    non_asymp_count = 0

    asymp_time_list = []
    
    if args.fname is not None:
        print("reading graph from", args.fname)
        with open(args.fname, 'r') as f:
            # needs to be just a graph exported from tracy's code
            res = [{k: np.asarray(v) for k, v in r.items()} for r in json.load(f)]

    #value settings:
    num_iter = args.max_iter #10000
    num_node = args.size #15
    simple_edge = True
    dynamic = False

    for runs in range(num_runs):
        
        #1st argument: number of Nodes


        # g = two_opposing_groups(50, 50, 1, 1, 1)
        #g = add_self_loops(g)
        coins = None
        coins_len = num_iter
        if args.fname is not None:
            res_run = res[runs]

            g = nx.from_numpy_matrix(res_run["adjacency"])
            g = set_friendliness_from_nparray(g, res_run["friendliness"])
            g = set_prior_from_nparray(g, res_run["initial_distr"])
            coins = res_run["coins"].tolist()

            coins_len = len(coins)
        else:
            g = nx.complete_graph(num_node)
            g = randomise_prior(g, mean=(0.0, 1.0), fwhm=(0.2, 0.8), bias_len=21)
            g = randomise_edge_weight(g, edge_weight=(-1, 1), simple_edge_weight=True, include_zero=True)

        fraction = 1/nx.number_of_edges(g)

        # g = generate_graph(number_of_nodes=2)
        


        # graph_evol = graph_evolution(g, f=0.1, num_iter=num_iter)
        #
        # print(graph_evol[0]==graph_evol[1])
        # print(graph_evol[0] is graph_evol[2])

        # g = recreate_graph('output/test', prior_info=0)
        # for u,v in g.edges:
        #     g.edges[u,v]['correlation'] = -1

        simulation = Simulation(g, true_bias=args.bias, tosses_per_iteration=args.tosses, output_direc='output/test_' + str(runs), bias_len=21, coin_obs_file=None, coins=coins)

        # simulation.do_simulation_without_signal(num_iter=100, learning_rate=0.4, blend_method=1)
        asymp_t = simulation.do_simulation(f=fraction, num_iter=num_iter, blend_method=0, learning_rate=args.learning_rate, dynamic=dynamic, simple_edge_weight=simple_edge)
        #print(nx.number_of_edges(g))

        print("Run {}/{}: Asymptotic Learning time: {}".format(runs+1, num_runs, asymp_t))

        if asymp_t:
            asymp_time_list.append(asymp_t)

            if coins is not None and asymp_t != coins_len:
                print("!! mismatch in cointosses used: (given/used) ", coins_len, "/", asymp_t)
        else:
            non_asymp_count+=1
            
            if coins is not None and coins_len != num_iter:
                print("!! mismatch in cointosses used: (given/used) ", coins_len, "/", num_iter)

    avg_asymp_time = np.mean(asymp_time_list)
    med_asymp_time = np.median(asymp_time_list)

    with open('n{}f{:.2f}_asymp_time.txt'.format(num_node, fraction), 'a') as f:
        f.write('{}\t {} \t {} \t {} \t {} \t {} \t {} \n'.format(num_node, fraction, simple_edge, non_asymp_count, avg_asymp_time, med_asymp_time, asymp_time_list))

    print("Average asymptotic time: {} timesteps".format(avg_asymp_time))
    print("Median asymptotic time: {} timesteps".format(med_asymp_time))

    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')


if __name__ == '__main__':
    main(parse_args())


# TODO allow for the signal appear at every n iteration, instead of every iteration
