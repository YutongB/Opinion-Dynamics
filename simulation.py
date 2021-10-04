import networkx as nx
import numpy as np
import scipy.stats
from read_graph import remove_files, recreate_graph, create_output_directory, read_coin_observations
import timeit
from generate_graph import two_opposing_groups, unbalanced_triangle, visualise_graph
import matplotlib.pyplot as plt
import os, sys
import pandas as pd


"""
Main purpose is run the automaton and store the prior at each iteration. The priors can then be fed to analyse.py
to produce relevant plots.

If you want to run an ensemble of simulations to collect statistics, better to use simulation_numba.py 
"""



class Simulation:
    def __init__(self, graph, true_bias=0.6, tosses_per_iteration=1, output_direc='output/default', coin_obs_file=None, bias_len=21, asymp_time_threshold=100):
        """
        Supply coin_obs_file with an output folder to reuse the same sequence of coin tosses
        Alternatively, directly assign the sequence of coin tosses to self.heads_list and self.tosses_list
        """
    
        # TODO remove bias_len and infer it from the length of the prior
        self.g = nx.Graph(graph)
        self.true_bias = true_bias
        self.tosses_per_iteration = tosses_per_iteration
        self.output_direc = output_direc
        self.bias_list = np.linspace(0, 1, bias_len)
        self.asymp_time_threshold = asymp_time_threshold
        self.beliefs_control = np.ones(bias_len, dtype=float) / bias_len

        if coin_obs_file is None:
            self.heads_list = []
            self.tosses_list = []
        else:
            if coin_obs_file is True:
                self.heads_list, self.tosses_list = read_coin_observations('output/default/data/observations.txt')
            else:
                if not coin_obs_file.endswith('.txt'):
                    coin_obs_file += '/data/observations.txt'
                self.heads_list, self.tosses_list = read_coin_observations(coin_obs_file)

        create_output_directory(output_direc)
        for node in self.g:
            self.g.nodes[node]['asymp_time'] = [0,-1, -1]  # (num consecutive t in asymp, mode, apparent mode)
            self.g.nodes[node]['old prior'] = self.g.nodes[node]['prior'].copy()


    def simulate_coin_toss(self):
        return np.random.binomial(self.tosses_per_iteration, self.true_bias)


    def calculate_coin_probability(self, num_heads, num_tosses):
        return scipy.stats.binom.pmf(num_heads, num_tosses, self.bias_list)


    def observe_coin(self):
        # observe coin once and update the priors
        num_heads = self.simulate_coin_toss()
        prob = self.calculate_coin_probability(num_heads, self.tosses_per_iteration)
        for node in self.g:
            norm = np.sum(self.g.nodes[node]['prior'] * prob)
            self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob / norm
            
        norm = np.sum(self.beliefs_control * prob)
        self.beliefs_control = self.beliefs_control * prob / norm
        
        return num_heads, self.tosses_per_iteration


    def blend_pos(self, learning_rate=0.25, method=0):
        """
        method == 0 is the method used in the paper. The others are for testing.
        """
        if method == 0:  # weighted average
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
                
                if total_edge_weight > 0:
                    if isinstance(learning_rate, (list, tuple, np.ndarray)):
                        new_pos *= learning_rate[i] / total_edge_weight
                    else:
                        new_pos *= learning_rate / total_edge_weight
                self_pos = np.maximum(0, self_pos + new_pos)
                self_pos /= self_pos.sum()
                self.g.nodes[i]['prior'] = self_pos

        elif method == 1:
            for i in self.g:
                # bayesian-like, double norm. This is what I've sent to andrew
                self_pos = self.g.nodes[i]['self_pos'].copy()
                for j in self.g.neighbors(i):
                    base_pos = np.ones(len(self.bias_list))
                    base_pos /= base_pos.sum()
                    edge_weight = self.g.edges[i, j]['correlation']
                    neighbour_pos = self.g.nodes[j]['self_pos']
                    if edge_weight < 0:
                        neighbour_pos = 1 - neighbour_pos
                        neighbour_pos /= neighbour_pos.sum()
                        
                    neighbour_pos = neighbour_pos + base_pos

                    neighbour_pos /= neighbour_pos.sum()
                    
                    self_pos *= neighbour_pos
                self_pos /= self_pos.sum()
                
                self.g.nodes[i]['prior'] = self_pos
        
        elif method == 2:
            for i in self.g:
                # bayesian-like
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
                    
                    self_pos *= neighbour_pos
                self_pos /= self_pos.sum()
                
                self.g.nodes[i]['prior'] = self_pos
                
        elif method == 3:
            # convolution
            print('a')
            for i in self.g:
                self_pos = self.g.nodes[i]['self_pos'].copy()
                for j in self.g.neighbors(i):
                    # TODO implement learning rate. For now, assume no effect from learning rate
                    self_pos = np.convolve(self_pos, self.g.nodes[j]['self_pos'], mode='valid')
                    
                self_pos /= self_pos.sum()
                self.g.nodes[i]['prior'] = self_pos


    def write_prior(self):
        # save the new prior
        for i in self.g:
            with open(f'{self.output_direc}/data/node{i}.txt', 'a') as f:
                for data in self.g.nodes[i]['prior']:
                    f.write(str(data) + ' ')
                f.write('\n')
                
        with open(f'{self.output_direc}/data/control.txt', 'a') as f:
            for data in self.beliefs_control:
                f.write(str(data) + ' ')
            f.write('\n')


    def save_graph(self):
        # save the priors into a new text file
        remove_files(f'{self.output_direc}/data', starts_with='node', ends_with='.txt')
        for node, prior in self.g.nodes(data='prior'):
            with open(f'{self.output_direc}/data/node{node}.txt', 'w+') as f:
                for i in prior:
                    f.write(str(i) + ' ')
                f.write('\n')
                
        with open(f'{self.output_direc}/data/control.txt', 'w+') as f:
            for i in self.beliefs_control:
                f.write(str(i) + ' ')
            f.write('\n')

        # save information about the graph
        os.makedirs(f'{self.output_direc}/data/adj', exist_ok=True)
        remove_files(f'{self.output_direc}/data/adj', starts_with='', ends_with='.txt')
        self.save_adjacency_matrix(0)


    def save_tosses(self):
        with open(f'{self.output_direc}/data/observations.txt', 'w+') as f:
            f.write('True coin bias: {}\n'.format(self.true_bias))
            f.write('Iteration Num_heads Num_tosses\n')
            for i in range(len(self.heads_list)):
                f.write(f'{i+1} {self.heads_list[i]} {self.tosses_list[i]}\n')
                
        with open(f'{self.output_direc}/data/asymp.txt', 'w+') as f:
            for i in self.get_proper_asymp_time_unsorted:
                f.write(f'{i}\n')
            
                
    def save_adjacency_matrix(self, t):
        with open(f'{self.output_direc}/data/adj/{t}.txt', 'w+') as f:
            f.write('Number of nodes: {}\n'.format(self.g.number_of_nodes()))
            for u, v, correlation in self.g.edges.data('correlation'):
                f.write('{} {} {}\n'.format(u, v, correlation))
                

    def update_node_asymp(self, threshold=0.01):
        """
        Updates each node's 'asymp time' parameter and returns the number of nodes that asymps
        """
        num_asymp = 0
        for node in self.g:
            if np.all(np.abs(self.g.nodes[node]['prior'] - self.g.nodes[node]['old prior']) <= threshold * np.max(self.g.nodes[node]['old prior'])):
                num_asymp += 1
                if self.g.nodes[node]['asymp_time'][0] == 0:
                    self.g.nodes[node]['asymp_time'][1] = np.argmax(self.g.nodes[node]['prior']) / (len(self.bias_list) - 1)
                    self.g.nodes[node]['asymp_time'][2] = np.argmax(self.beliefs_control) / (len(self.bias_list) - 1)
                self.g.nodes[node]['asymp_time'][0] += 1
            else:
                self.g.nodes[node]['asymp_time'] = [0, -1, -1]  # (num consecutive t in asymp, mode, apparent mode)
                self.g.nodes[node]['old prior'] = self.g.nodes[node]['prior'].copy()
        
        return num_asymp
        

    def do_simulation(self, num_iter=10, learning_rate=0.25, blend_method=0):
        """
        Runs the automaton and returns the number of time steps to asymp, 0 if did not asymp within num_iter
        """
        self.save_graph()
        
        t = 1
        asymptotic_learning_time = 0

        if len(self.heads_list) > 0:
            # heads/tosses already determined, so follow that
            print('Using predetermined coin tosses')
            while t <= len(self.heads_list):
            # for t in range(len(self.heads_list)):
                if t % 100 == 0:
                    print('Iteration {}'.format(t), end='\r')
                prob = self.calculate_coin_probability(self.heads_list[t-1], self.tosses_list[t-1])
                for node in self.g:
                    norm = np.sum(self.g.nodes[node]['prior'] * prob)
                    self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob / norm
                    
                norm = np.sum(self.beliefs_control * prob)
                self.beliefs_control = self.beliefs_control * prob / norm

                self.blend_pos(learning_rate=learning_rate, method=blend_method)
                self.write_prior()
                    
                if self.update_node_asymp() == self.g.number_of_nodes():
                    asymptotic_learning_time += 1
                    if asymptotic_learning_time >= self.asymp_time_threshold:
                        print(f'Reached asymptotic learning at iteration {t}.')
                        self.heads_list = self.heads_list[:t]
                        self.tosses_list = self.tosses_list[:t]
                        self.save_tosses()
                        return t
                else:
                    asymptotic_learning_time = 0
                t += 1
                
            if t > len(self.heads_list) and t <= num_iter:
                print('')
                print('All predetermined coin tosses used. Will now generate new tosses')

        while t <= num_iter:
            # simulate our own coin tosses.
            if t % 100 == 0:
                print('Iteration {}'.format(t), end='\r')
            num_heads, num_tosses = self.observe_coin()
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            self.heads_list.append(num_heads)
            self.tosses_list.append(num_tosses)
            self.write_prior()
            if self.update_node_asymp() == self.g.number_of_nodes():

                asymptotic_learning_time += 1
                if asymptotic_learning_time >= self.asymp_time_threshold:
                    print('')
                    print(f'Reached asymptotic learning at iteration {t}.')
                    self.save_tosses()
                    return t
            else:
                asymptotic_learning_time = 0
            t += 1
            
        print('')
        self.save_tosses()
        return 0

    def do_simulation_without_signal(self, num_iter=10, learning_rate=0.25, blend_method=0):
        """
        Same as do_simulation, but there is no coin tosses
        """
        self.heads_list = []
        self.tosses_list = []
        self.save_graph()
        
        t = 1
        asymptotic_learning_time = 0
        
        while t <= num_iter:
            if t % 100 == 0:
                print('Iteration {}'.format(t))
            for i in self.g:
                self.g.nodes[i]['self_pos'] = self.g.nodes[i]['prior'].copy()

            self.heads_list.append(0)
            self.tosses_list.append(0)
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            self.write_prior()
            if self.update_node_asymp() == self.g.number_of_nodes():

                asymptotic_learning_time += 1
                if asymptotic_learning_time >= self.asymp_time_threshold:
                    print(f'Reached asymptotic learning at iteration {t}.')
                    self.save_tosses()
                    return t
            else:
                asymptotic_learning_time = 0
            
            t += 1

        self.save_tosses()
        
    def do_simulation_without_print_and_save(self, num_iter=1000, learning_rate=0.25, blend_method=0):
        """
        Same as do_simulation, except we don't write any information.
        simulation_numba.py now exists, so use that instead for coputational speed
        
        This function is still here so nothing in engine.py breaks for now
        """
        
        t = 1
        asymptotic_learning_time = 0

        if len(self.heads_list) > 0:
            # heads/tosses already determined, so follow that
            while t <= len(self.heads_list):
                prob = self.calculate_coin_probability(self.heads_list[t-1], self.tosses_list[t-1])
                for node in self.g:
                    # norm = np.sum(self.g.nodes[node]['prior'] * prob)
                    # self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob / norm
                    self.g.nodes[node]['self_pos'] = self.g.nodes[node]['prior'] * prob 
                    self.g.nodes[node]['self_pos'] = self.g.nodes[node]['self_pos'] / self.g.nodes[node]['self_pos'].sum()
                    
                norm = np.sum(self.beliefs_control * prob)
                self.beliefs_control = self.beliefs_control * prob / norm

                self.blend_pos(learning_rate=learning_rate, method=blend_method)
                if self.update_node_asymp() == self.g.number_of_nodes():
                    asymptotic_learning_time += 1
                    if asymptotic_learning_time >= self.asymp_time_threshold:
                        self.heads_list = self.heads_list[:t]
                        self.tosses_list = self.tosses_list[:t]
                        return t
                else:
                    asymptotic_learning_time = 0
                t += 1

        while t <= num_iter:
            # simulate our own coin tosses.
            num_heads, num_tosses = self.observe_coin()
            self.heads_list.append(num_heads)
            self.tosses_list.append(num_tosses)
            self.blend_pos(learning_rate=learning_rate, method=blend_method)
            if self.update_node_asymp() == self.g.number_of_nodes():
                asymptotic_learning_time += 1
                if asymptotic_learning_time >= self.asymp_time_threshold:
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
            if self.update_node_asymp() == self.g.number_of_nodes():
                print(f'Reached asymptotic learning at iteration {t}.')
                break

    """
    These functions should really belong to analyse.py
    """

    @property
    def get_width(self):
        # find out how far the distribution of belief tilts away from the true bias. Also finds out if there is imbalance between the widths
        # note: max_lower_width is a negative number if the node's belief is smaller than the true bias
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
        
    
    @property
    def do_all_nodes_agree(self):
        # True if all nodes have the same posteriors. To cheat, only check if their argmax are equal
        for node in range(1, self.g.number_of_nodes()):
            if np.argmax(self.g.nodes[node]['prior']) != np.argmax(self.g.nodes[0]['prior']) :
                return False
        return True
        
        
    def number_of_correct_nodes(self, opinion_range=None, criteria='mode'):
        # returns the number of nodes with modes inside the opinion_range (bounds inclusive)
        # possible criteria: 'mode' (default), 'mean'
        if opinion_range is None:
            opinion_range = (0.45, 0.55)
        if criteria == 'mean':
            criteria = lambda x: np.sum(x * self.bias_list)
        else:  #default is mode
            criteria = lambda x: np.argmax(x)  / (len(self.bias_list) - 1)
        num_correct = 0
        for node in self.g.nodes():
            if opinion_range[0] <= criteria(self.g.nodes[node]['prior']) <= opinion_range[1]:
                num_correct += 1
        return num_correct

    
    @property
    def get_asymp_time(self):
        # returns the num of consecutive t that each node spends in asymp learning, in ascending order.
        # this means that the first entry is the last node to achieve asymp learning
        data = [i[1] for i in self.g.nodes(data='asymp_time')]  # i[0] here is node label, which we don't want
        data.sort(key=lambda x: x[0])
        return data
    
    
    @property
    def get_proper_asymp_time(self):
        data = self.get_asymp_time
        asymp_time = len(self.heads_list)
        return [[asymp_time - i[0], i[1], i[2]] for i in data]
        
    @property
    def get_proper_asymp_time_unsorted(self):
        data = [i[1] for i in self.g.nodes(data='asymp_time')]
        asymp_time = len(self.heads_list)
        return [[asymp_time - i[0], i[1], i[2]] for i in data]
        
        
    @property
    def is_first_node_correct(self):
        return self.get_asymp_time[-1][1] == self.get_asymp_time[-1][2]
        
    
    @property
    def get_apparent_true_bias(self):
        return self.beliefs_control


def put_smaller_number_first(number):
    return min(number), max(number)


def resized_range(number):
    return np.random.random() * (number[1] - number[0]) + number[0]


def gaussian(bias_list, mean=0.5, fwhm=0.5):
    not_norm = np.exp(-4 * np.log(2) * ((bias_list - mean) / fwhm) ** 2)
    return not_norm / not_norm.sum()
    
    
def gaussian_stddev(bias_list, mean=0.5, stddev=0.25):
    not_norm = np.exp(-((bias_list-mean) / stddev)**2 / 2)
    return not_norm / not_norm.sum()
    

def fwhm_to_std(fwhm):
    factor = 2* np.sqrt(2* np.log(2))
    return fwhm / factor
    

def uniform_distribution(bias_len=21):
    return np.ones(bias_len, dtype=float) / bias_len

    
def make_evenly_spaced_priors(g, fwhm=0.4, bias_len=21):
    bias_list = np.linspace(0, 1, bias_len)
    mean_list = np.linspace(0, 1, g.number_of_nodes())
    
    for count, mu in enumerate(mean_list):
        g.nodes[count]['prior'] = gaussian(bias_list, mean=mu, fwhm=fwhm)
        
    return g


def randomise_prior(g, mean=(0.0, 1.0), stddev=(0.1, 0.4), bias_len=21):
    g = nx.Graph(g)
    bias_list = np.linspace(0, 1, bias_len)
    mean = put_smaller_number_first(mean)
    stddev = put_smaller_number_first(stddev)

    for i in range(g.number_of_nodes()):
        # randomise initial prior of each node. assume normal distribution.
        mean1 = resized_range(mean)
        stddev1 = resized_range(stddev)
        prior = gaussian_stddev(bias_list, mean1, stddev1)
        prior /= prior.sum()
        g.nodes[i]['prior'] = prior

    return g


def randomise_edge_weight(g, edge_weight=(-1, 1), simple_edge_weight=True):
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
            elif roll == 0.0:
                # should never happen, but it's here just in case
                roll = 1
        g.edges[i, j]['correlation'] = roll

    return g


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


class HiddenPrints:
    """
    This just hides all print statements
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def extend_simulation(input_folder, output_folder='', t=0):
    """
    Given a simulation folder, extends the simulation with identical priors and coin tosses up to t
    """
    g = recreate_graph(input_folder, 0)
    if not output_folder:
        output_folder = f'{input_folder}_extended{t}'
    
    simulator = Simulation(g, true_bias=0.6, tosses_per_iteration=1, output_direc=output_folder, coin_obs_file=input_folder, bias_len=21, asymp_time_threshold=t)
    
    if t == 0:
        t = len(simulator.heads_list) * 10
    simulator.asymp_time_threshold = t
    simulator.do_simulation(num_iter=t, blend_method=0, learning_rate=0.25)
    

def main():
    repetitions = 10
    time_list = []
    
    for i in range(repetitions):
        if i % 1 == 0:
            print(i)
        g = nx.complete_graph(10)
        g = randomise_prior(g, bias_len=21)
        g = randomise_edge_weight(g, edge_weight=(-1, 1))
        simulation = Simulation(g, true_bias=0.5, tosses_per_iteration=10, output_direc='output/test', bias_len=21, coin_obs_file=None, asymp_time_threshold=1000)
        with HiddenPrints():
            time1 = timeit.default_timer()
            simulation.do_simulation_without_print_and_save(num_iter=1000, blend_method=0, learning_rate=0.25)
            time2 = timeit.default_timer()
        time_list.append(time2 - time1)
    
    df = pd.DataFrame({'Result': time_list})
    print(df.describe())


if __name__ == '__main__':
    g = recreate_graph('output/ba3')
