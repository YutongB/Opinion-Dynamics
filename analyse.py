import numpy as np
import scipy as sp
import scipy.stats
import scipy.stats.distributions as distributions
from scipy.optimize import curve_fit
import networkx as nx
import itertools
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator  # to set plot ticks to integers
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, Slider
from read_graph import extract_basic_info, remove_files, extract_prior, recreate_graph, extract_true_coin_bias, extract_prior_all_time, read_coin_observations, get_asymp_info
import timeit
from PIL import Image
import glob
import natsort as ns
import os
import pandas as pd


# TODO extract_prior() is very inefficient, because opening a file is costly. Replace with something that opens one file only once


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


class DataAnalyser:

    def __init__(self, direc='output/default'):
        self.direc = direc
        self.file_name_prepend = ''
        self.summary_stats_dict = {}

    def calculate_summary_statistics(self, control=0):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)

        mean_list = np.zeros((len(time_list), number_of_nodes))
        mode_list = np.zeros((len(time_list), number_of_nodes))
        stddev_list = np.zeros((len(time_list), number_of_nodes))
        param_dict = {'mean': mean_list, 'mode': mode_list, 'stddev': stddev_list}  # for looping the parameters
        
        for node in range(number_of_nodes):
            with open(f'{self.direc}/data/node{node}.txt', 'r') as f:
                lines = f.readlines()
                for t in time_list:
                    prior = np.array([float(data) for data in lines[t].split()])
                    mean = np.sum(prior * bias_list)
                    stddev = np.sqrt(sum((bias_list - mean) ** 2 * prior))
                    mode = np.argmax(prior) / (len(bias_list) - 1)
                    
                    mean_list[t, node] = mean
                    stddev_list[t, node] = stddev
                    mode_list[t, node] = mode
                    
        control_data_dict = {'mean': np.zeros(len(time_list)), 'mode': np.zeros(len(time_list)), 'stddev': np.zeros(len(time_list))}
        control_beliefs = self.get_control_beliefs(node=control)
        for t in time_list:
            prior = control_beliefs[t]
            mean = np.sum(prior * bias_list)
            stddev = np.sqrt(sum((bias_list - mean) ** 2 * prior))
            mode = np.argmax(prior) / (len(bias_list) - 1)
            
            control_data_dict['mean'][t] = mean
            control_data_dict['mode'][t] = mode
            control_data_dict['stddev'][t] = stddev
                
        # with open(f'{self.direc}/data/control.txt', 'r') as f:
        #     lines = f.readlines()
        #     for t in time_list:
        #         prior = np.array([float(data) for data in lines[t].split()])
        #         mean = np.sum(prior * bias_list)
        #         stddev = np.sqrt(sum((bias_list - mean) ** 2 * prior))
        #         mode = np.argmax(prior) / (len(bias_list) - 1)
        #         
        #         control_data_dict['mean'].append(mean)
        #         control_data_dict['mode'].append(mode)
        #         control_data_dict['stddev'].append(stddev)
        
        self.summary_stats_dict = param_dict
        return param_dict, control_data_dict
        
    def get_control_beliefs(self, node=0):
        _, time_list, bias_list = extract_basic_info(self.direc)
        belief_list = np.zeros((len(time_list), len(bias_list)))
        with open(f'{self.direc}/data/node{node}.txt', 'r') as f:
            line = f.readline().rstrip()
            belief_list[0] = np.array([float(data) for data in line.split()])
            
        heads_list, _ = read_coin_observations(f'{self.direc}/data/observations.txt')
        
        for t in time_list[:-1]:
            if heads_list[t] == 1:
                new_belief = belief_list[t] * bias_list
            else:
                new_belief = belief_list[t] * (1-bias_list)
            new_belief = new_belief / new_belief.sum()
            belief_list[t+1] = new_belief
            
        return belief_list      
        

    def produce_plots(self):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        num_iter = len(time_list)
        control = 2

        if num_iter > 10:
            time_spacing = num_iter // 100 * 5  # every 100 t, show only 5 of them
            if time_spacing == 0:
                time_spacing = 5
            chopped_time_list = [int(i) for i in list(set().union(range(10), range(0, num_iter, time_spacing)))]
            if num_iter-1 not in chopped_time_list:
                chopped_time_list.append(num_iter - 1)

        else:
            chopped_time_list = time_list

        # overplot the posteriors
        remove_files(f'{self.direc}/prior', ends_with='.png')
        for t in chopped_time_list:
            fig, ax = plt.subplots()
            for i in range(number_of_nodes):
                prior = extract_prior(self.direc, node=i, t=t)
                if number_of_nodes < 8:
                    ax.plot(bias_list, prior, label=f'Agent {i+1}')
                else:
                    ax.plot(bias_list, prior)

            ax.set_xlabel('Bias')
            ax.set_xlabel(r'$\theta$')
            ax.set_ylabel('Probability')
            ax.grid(linestyle='--', axis='both')
            if t == 0:
                ax.set_title('Initial beliefs')
            else:
                ax.set_title(r'$t$ = {}'.format(t))

            if number_of_nodes < 8:
                ax.legend()
            # plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
            plt.tight_layout()
            fig.savefig(f'{self.direc}/prior/{self.file_name_prepend}t={t}.png')
            plt.close()
            
            if t == 0:
                fig, ax = plt.subplots()
                for i in range(number_of_nodes):
                    prior = extract_prior(self.direc, node=i, t=t)
                    ax.plot(bias_list, prior*sp.stats.binom.pmf(5, 10, bias_list) / np.sum(prior * sp.stats.binom.pmf(5, 10, bias_list)), label=str(i))
                    
                # ax.set_xlabel('Bias')
                # ax.set_xlabel(r'$\theta$')
                # ax.set_ylabel('Probability')
                # ax.grid(linestyle='--', axis='x')
                # plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
                # plt.tight_layout()
                # fig.savefig(f'{self.direc}/prior/{self.file_name_prepend}ideal.png')
                # plt.close()

        # calculate mean, mode and std dev
        param_dict, control_dict = self.calculate_summary_statistics(control)

        for param_name, data in param_dict.items():
            remove_files(f'{self.direc}/{param_name}', ends_with='.png')

        for t in time_list:
            # also plot histograms of the parameters at a certain time
            if t in chopped_time_list:
                for pair in param_dict.items():
                    param_name = pair[0]
                    data = pair[1][t]

                    fig, ax = plt.subplots()
                    ax.hist(data, bins=np.arange(np.min(data)-0.025, np.max(data) + 0.05, 0.05))
                    ax.set_xlabel(param_name.capitalize())
                    ax.set_ylabel('Count')
                    ax.set_title(r'$t$={}'.format(t))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # set the y ticks to integers only
                    ax.grid(linestyle='--', axis='both')
                    plt.tight_layout()
                    fig.savefig('{}/{}/t={}'.format(self.direc, param_name, t))
                    plt.close()

        # time series - boxplot
        remove_files(f'{self.direc}/time_series', ends_with='.png')

        # time series - boxplot. boxplots only worth it for enough data points
        # TODO separate plots if time_list is too big
        if number_of_nodes > 10 and len(time_list) < 20:
            for param_name, data in param_dict.items():
                fig, ax = plt.subplots()
                ax.boxplot(data, positions=time_list)  # TODO figure out how to show certain ticks only
                ax.set_xlabel('Iteration')
                ax.set_ylabel(param_name.capitalize())
                ax.grid(linestyle='--')
                plt.tight_layout()
                fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}box_{param_name}.png')
                plt.close()

        # draw the network. Above 20 nodes, don't do this because it gets too messy
        if 2 < number_of_nodes < 20:
            mean_list = param_dict['mean']
            g = recreate_graph(self.direc)
            edge_data = [x[2] for x in g.edges.data('correlation')]

            # edge_data = [x[2] for x in g.edges.data('correlation')]
            layout = nx.spring_layout(g)  # to keep the position of the nodes consistent between iterations

            remove_files(f'{self.direc}/network', ends_with='.png')
            
            is_dynamic = os.path.isfile(f'{self.direc}/data/adj/1.txt')
            for t in chopped_time_list:
                if t > 0 and is_dynamic:
                    with open(f'{self.direc}/data/adj/{t}.txt', 'r') as f:
                        lines = f.readlines()
                        del lines[0]
                        edge_data = [float(line.split()[-1]) for line in lines]
            
                fig, ax = plt.subplots()
                ax.set_title('Iteration {}'.format(t))

                nx.draw(g, pos=layout, ax=ax, node_color=mean_list[t], cmap='Blues', with_labels=True,
                        edge_color=edge_data, edge_vmin=-1, edge_vmax=1, edge_cmap=plt.cm.get_cmap('bwr_r'))

                cbar = plt.colorbar(
                    plt.cm.ScalarMappable(plt.Normalize(min(mean_list[t]), max(mean_list[t])), cmap='Blues'))
                cbar.set_label('Nodes: bias mean value', rotation=270, labelpad=10)
                cbar = plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(-1, 1), cmap='bwr_r'), ax=[ax], location='left')
                cbar.set_label('Edges: weight')

                # fig.tight_layout()  # axes supposedly incompatible with tight layout
                fig.savefig(f'{self.direc}/network/{self.file_name_prepend}t={t}.png')
                plt.close()

        # transpose the data. now the first index labels the node, the second labels time
        new_dict = {}
        for param, data in param_dict.items():
            new_dict[param] = np.transpose(data)
        param_dict = new_dict
        
        # color_cycle = plt.rcParams["axes.prop_cycle"].by_key()['color']
        for param_name, data in param_dict.items():
            fig, ax = plt.subplots()
            # ax.plot(time_list, control_dict[param_name], label='Control', ls='--', color=color_cycle[-3])
            #ax.plot(time_list, control_dict[param_name], label='Control', ls='--')
            for node, node_data in enumerate(data):
                #ax.plot(time_list, node_data, label=f'Agent {node+1}', color=color_cycle[node])
                ax.plot(time_list, node_data, label=f'Agent {node+1}')
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(param_name.capitalize())
            if param_name == 'mean':
                ax.set_ylabel(r'$\langle \theta \rangle$')
            elif param_name == 'stddev':
                ax.set_ylabel(r'$\sqrt{\langle(\theta - \langle \theta \rangle)^2 \rangle}$')
            
            if param_name != 'stddev':
                start, end = ax.get_ylim()
                first = np.ceil(start*10) / 10
                last = np.ceil(end*10) / 10
                ax.set_yticks(np.arange(first, last, 0.1))
                
            else:
                pass
                # params = self.fit_power_law()
                # label = r'$\propto t ^ {' + f'{params[1]:.2f}' + r'}$'
                # plot_time = 3
                # ax.plot(time_list[plot_time:], params[0] * time_list[plot_time:] ** params[1], label=label, ls='--')
                # ax.plot(time_list[plot_time:], params[0] * time_list[plot_time:] ** params[1], label=label, ls='--', color=color_cycle[number_of_nodes])
                # ax.set_ylabel('Standard deviation')
            
            if number_of_nodes < 8:
                ax.legend()
            
            ax.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}lines_{param_name}.png')
            plt.close()
            
            midpoint = len(time_list) // 2
            fig, ax = plt.subplots()
            for node, node_data in enumerate(data):
                ax.plot(time_list[midpoint:], node_data[midpoint:], label=str(node))
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param_name.capitalize())
            
            if param_name != 'stddev':
                start, end = ax.get_ylim()
                first = np.ceil(start*10) / 10
                last = np.ceil(end*10) / 10
                ax.set_yticks(np.arange(first, last, 0.1))
                
            else:
                ax.set_ylabel('Standard deviation')
            
            if number_of_nodes < 8:
                ax.legend()
            
            ax.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}lines_chopped_{param_name}.png')
            plt.close()
            

    def find_convergence(self):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        average_stddev_list = []
        convergent_threshold = 0.01
        reached_total_convergence = False
        first_convergence = -1
        total_convergence_time = 0
        remove_files(f'{self.direc}/converge')

        with open(f'{self.direc}/converge/{self.file_name_prepend}converge.txt', 'w+') as f:
            f.write('Nodes that share a letter have converged onto each other under a threshold of {}\n\n'.format(
                convergent_threshold))
            f.write('Node Letter\n')

            # store info of stddev of difference
            with open(f'{self.direc}/converge/{self.file_name_prepend}stddev.txt', 'w+') as f2:
                for t in time_list:
                    f.write(f'Time: {t}\n')
                    f2.write(f'Time: {t}\n')

                    # finding common letters to label nodes that have converged is a problem in finding maximal cliques.
                    # Hence, build a second graph whose connections indicate that the nodes have converged.
                    g2 = nx.Graph()
                    for n in range(number_of_nodes):
                        g2.add_node(n)

                    total_stddev = 0  # to find the average stddev for a particular t
                    count = 0

                    # iterate over all unique pairs of nodes
                    for u, v in itertools.combinations(range(number_of_nodes), r=2):
                        prior1 = extract_prior(self.direc, node=u, t=time_list[t])
                        prior2 = extract_prior(self.direc, node=v, t=time_list[t])

                        diff_stddev = np.std(prior1 - prior2)
                        if diff_stddev < convergent_threshold:
                            g2.add_edge(u, v)

                        total_stddev += diff_stddev
                        count += 1
                        f2.write(f'{u} {v} {diff_stddev}\n')

                    average_stddev_list.append(total_stddev / count)

                    cliques = list(nx.find_cliques(g2))

                    max_space = np.floor(
                        np.log10(number_of_nodes))  # to ensure numbers with different digits are aligned
                    node_list = [str(i) + '   ' + ' ' * int(max_space - (np.floor(np.log10(i)) if i > 0 else 0)) for i
                                 in range(number_of_nodes)]

                    if len(cliques) == 1 and not reached_total_convergence:
                        print(f'Time: {t}, all nodes have converged')
                        reached_total_convergence = True

                    elif len(cliques) > 1 and reached_total_convergence:
                        print(f'Time: {t}, nodes have deviated from convergence.')
                        reached_total_convergence = False

                    if reached_total_convergence and t > 0:
                        if first_convergence == -1:
                            first_convergence = t
                        total_convergence_time += 1

                    current_letter = ord('a')
                    number_of_primes = 0  # if there are more than 52 characters, annonate them with primes instead
                    for clique in cliques:
                        for node in range(number_of_nodes):
                            if node in clique:
                                node_list[node] += ' ' + chr(current_letter) + "'" * number_of_primes
                            else:
                                node_list[node] += '  ' + ' ' * number_of_primes

                        if current_letter == ord('z'):
                            current_letter = ord('A')
                        elif current_letter == ord('Z'):
                            current_letter = ord('a')
                            number_of_primes += 1
                        else:
                            current_letter += 1

                    for line in node_list:
                        f.write(line + '\n')
                    f.write('\n')

                f2.write('\nAverage stddev\n')
                for count, stddev in enumerate(average_stddev_list):
                    f2.write(f'{count} {stddev}\n')

            f.write(f'\nFirst convergence: {first_convergence}\nTotal time converged: {total_convergence_time}')

    def graph_convergence(self):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        with open(f'{self.direc}/converge/{self.file_name_prepend}stddev.txt', 'r') as f:
            lines = f.readlines()[-len(time_list):]

        x = []
        y = []
        for line in lines:
            data = line.split()
            x.append(int(data[0]))
            y.append(float(data[1]))

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.hlines(0.01, 0, x[-1], colors='r', linestyles='--')
        ax.set_ylabel('Stddev')
        ax.set_xlabel('Time')
        ax.set_title('Convergence')
        fig.savefig(f'{self.direc}/converge/{self.file_name_prepend}threshold.png')
        plt.close()

    def calculate_true_bias_estimates(self):
        true_bias = extract_true_coin_bias(self.direc)
        if not self.summary_stats_dict:
            self.calculate_summary_statistics()

        # exclude first data
        mean_list = self.summary_stats_dict['mean'][1:]
        mode_list = self.summary_stats_dict['mode'][1:]
        correct_width = 0.05

        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        correct_mean_list = []
        correct_mode_list = []
        all_correct_mean = 0
        all_correct_mode = 0

        for t in range(len(mean_list)):
            mean = np.array(mean_list[t])
            correct_guess = len(np.where(np.logical_and(true_bias - correct_width <= mean, mean <= true_bias + correct_width))[0])
            correct_mean_list.append(correct_guess)
            if correct_guess == number_of_nodes:
                all_correct_mean += 1

            mode = np.array(mode_list[t])
            correct_guess = len(np.where(np.logical_and(true_bias - correct_width <= mode, mode <= true_bias + correct_width))[0])
            correct_mode_list.append(correct_guess)
            if correct_guess == number_of_nodes:
                all_correct_mode += 1

        print(f'On average {sum(correct_mean_list)/len(correct_mean_list):.2f} and {sum(correct_mode_list)/len(correct_mode_list):.2f} have correct mean/mode')
        print(f'All nodes are correct for {all_correct_mean} (mean) and {all_correct_mode} (mode) time steps (out of {time_list[-1]})')

    def save_priors_into_gif(self):
        input_path = f'{self.direc}/prior/t=*.png'
        output_path = f'{self.direc}/image.gif'

        img, *imgs = [Image.open(f) for f in ns.natsorted(glob.glob(input_path))]
        img.save(fp=output_path, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
        
    def find_width(self):
        # find out how far the distribution of belief tilts away from the true bias. Also finds out if there is imbalance between the widths
        true_bias = extract_true_coin_bias(self.direc)
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        max_upper_width = 0.0
        max_lower_width = 0.0
        
        for n in range(number_of_nodes):
            prior = extract_prior(self.direc, node=n, t=-1)
            max_arg = np.argmax(prior)
            bias = max_arg * bias_list[1]
            width = true_bias - bias
            
            if width > max_upper_width:
                max_upper_width = width
            elif width < max_lower_width:
                max_lower_width = width
            
        return max_upper_width, max_lower_width

    def fit_power_law(self):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        
        node = 0
        min_time = 50
        max_time = 500
        
        time_list = np.array(time_list)
        # time_list = time_list[50:500]
        stddev_list = np.zeros((number_of_nodes, len(time_list)))
        
        for node in range(number_of_nodes):
            with open(f'{self.direc}/data/node{node}.txt', 'r') as f:
                lines = f.readlines()
                for t in time_list:
                    prior = np.array([float(data) for data in lines[t].split()])
                    mean = np.sum(prior * bias_list)
                    stddev = np.sqrt(sum((bias_list - mean) ** 2 * prior))
                    
                    stddev_list[node, t - time_list[0]] = stddev
                    
        def power_law(x, amplitude, power):
            return amplitude * (x ** power)
            
        parameters, cov = curve_fit(power_law, time_list[min_time:max_time], stddev_list[0][min_time:max_time], bounds=(-np.inf, (np.inf, 0)))
        print(parameters)
        print(cov)
        stddev_error = np.sqrt(np.diag(cov))
        print(stddev_error)
        
        alpha = 0.05
        n = len(time_list)
        p = len(parameters)
        dof = max(0, n-p)
        tval = distributions.t.ppf(1.0-alpha/2.0, dof)
        
        lower_parameters = parameters - stddev_error*tval
        upper_parameters = parameters + stddev_error*tval
        print(r'95% confidence interval')
        print(lower_parameters)
        print(upper_parameters)
        
        # label = r'$\propto t ^ {' + f'{parameters[1]:.2f}' + r'}$'
        # plot_time = 4
        # 
        # fig, ax = plt.subplots()
        # ax.plot(time_list, stddev_list[0], label='Data')
        # # ax.plot(time_list, power_law(time_list, *parameters), label=r'$\propto t ^ {-0.51}$')
        # ax.plot(time_list[plot_time:], power_law(time_list[plot_time:], *parameters), label=label)
        # #ax.plot(time_list, power_law(time_list, *lower_parameters), label='Fit lower')
        # #ax.plot(time_list, power_law(time_list, *upper_parameters), label='Fit upper')
        # ax.legend()
        # ax.grid(linestyle='--', axis='both')
        # fig.tight_layout()
        # plt.show()
        return parameters

    def find_belief_convergence(self, for_all=False):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        
        if for_all:
            belief_list = np.zeros((number_of_nodes, len(time_list), 21))
            for n in range(number_of_nodes):
                belief_list[n] = extract_prior_all_time(self.direc, n)
            beliefs_diff = np.abs(belief_list - belief_list[0])
            max_beliefs = np.max(belief_list, axis=(0,2))
            
            threshold = 0.001
            a = np.all(beliefs_diff < threshold * max_beliefs[:, None], axis=(0,2))
            print(np.argmax(a))
                    
        else:
            
            beliefs1 = extract_prior_all_time(self.direc, 0)
            beliefs2 = extract_prior_all_time(self.direc, 1)
            
            control_beliefs = self.get_control_beliefs()
            
            beliefs_compare1 = beliefs2
            beliefs_compare2 = beliefs1
            
            beliefs_diff = np.abs(beliefs_compare1 - beliefs_compare2)
            max_beliefs = np.maximum(np.max(beliefs_compare1, axis=1), np.max(beliefs_compare2, axis=1))
            
            
            threshold = 0.001
            a = np.all(beliefs_diff < threshold * max_beliefs[:, None], axis=1)
            print(np.argmax(a))
        
    def find_belief_oscillation(self, node=2, asymp_time=1200):
        # 1117
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        print(len(time_list))
        
        belief_osc = extract_prior_all_time(self.direc, node)
        
        other_nodes = [i for i in range(3) if i != node]
        beliefs2 = extract_prior_all_time(self.direc,other_nodes[1])
        beliefs1 = extract_prior_all_time(self.direc,other_nodes[0])
        
        median1 = np.argmax(beliefs1[asymp_time])
        median2 = np.argmax(beliefs2[asymp_time])
        
        medians = (min(median1, median2), max(median1, median2))
        lower_median = [1.0, 0.0]
        upper_median = [1.0, 0.0]
        
        time_diff = []
        confidence_deviation = []
        lower_amplitude = []
        upper_amplitude = []
        max_confidence = 0.0
        lower_confidence = 0.0
        upper_confidence = 0.0
        last_time = asymp_time
        
        if belief_osc[asymp_time][medians[0]] > belief_osc[asymp_time][medians[0]]:
            last_higher_median = medians[0]
        else:
            last_higher_median = medians[1]
        
        for t, belief in enumerate(belief_osc[asymp_time:], asymp_time):
            lower_belief = belief[medians[0]]
            upper_belief = belief[medians[1]]
            if max_confidence == 0.0:
                max_confidence = np.abs(lower_belief - 0.5)
                lower_confidence = lower_belief - 0.5
                upper_confidence = upper_belief - 0.5
            
            if lower_belief < lower_median[0]:
                lower_median[0] = lower_belief
            if lower_belief > lower_median[1]:
                lower_median[1] = lower_belief
            if upper_belief < upper_median[0]:
                upper_median[0] = upper_belief
            if upper_belief > upper_median[1]:
                upper_median[1] = upper_belief
                
            if lower_belief > upper_belief:
                higher_median = medians[0]
            else:
                higher_median = medians[1]
                
            if higher_median != last_higher_median:
                last_higher_median = higher_median
                time_diff.append(t - last_time)
                last_time = t
                confidence_deviation.append(max_confidence)
                lower_amplitude.append(lower_confidence)
                upper_amplitude.append(upper_confidence)
                max_confidence = 0.0
                lower_confidence = 0.0
                upper_confidence = 0.0
            
            deviation = np.abs(0.5 - lower_belief)
            if deviation > max_confidence:
                max_confidence = deviation
                
            if np.abs(lower_confidence) < np.abs(lower_belief - 0.5):
                lower_confidence = lower_belief - 0.5
            if np.abs(upper_confidence) < np.abs(upper_belief - 0.5):
                upper_confidence = upper_belief - 0.5
            
        print(f'Lower median: {lower_median}')
        print(f'Upper median: {upper_median}')
        
        df = pd.DataFrame({'time_diff': time_diff, 'Amplitude': confidence_deviation, 'Lower deviation': lower_amplitude, 'Upper deviation': upper_amplitude})
        print(df.describe())
        
        fig, ax = plt.subplots()
        ax.hist(df['time_diff'], bins=np.arange(min(time_diff)-0.5, max(time_diff)+0.5, 1), rwidth=1)
        ax.set_xlabel('Switching period')
        ax.set_ylabel('Count')
        ax.grid(linestyle='--', axis='both')
        fig.tight_layout()
        
        # fig, ax = plt.subplots()
        # ax.hist(df['Amplitude'], bins='auto')
        # ax.set_xlabel('Oscillation amplitude')
        # ax.set_ylabel('Count')
        # ax.grid(linestyle='--', axis='both')
        # fig.tight_layout()
        
        fig, ax = plt.subplots()
        ax.hist(df['Lower deviation'], bins='auto')
        ax.set_xlabel(r'Oscillation amplitude ($\theta=\theta_0$)')
        ax.set_ylabel('Count')
        ax.grid(linestyle='--', axis='both')
        fig.tight_layout()
        
        # fig, ax = plt.subplots()
        # ax.hist(df['Upper deviation'], bins='auto')
        # ax.set_xlabel('Oscillation deviation (upper)')
        # ax.set_ylabel('Count')
        # ax.grid(linestyle='--', axis='both')
        # fig.tight_layout()
        
        print('')
        print(df.quantile(0.95))
        
        plt.show()
    
    def get_apparent_bias(self):
        heads_list, _ = read_coin_observations(f'{self.direc}/data/observations.txt')
        running_heads = np.cumsum(heads_list)
        running_tosses = np.arange(1, running_heads.shape[0]+1)
        percentage = running_heads / running_tosses
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(running_heads.shape[0])[10:], percentage[10:])
        ax.grid(linestyle='--', axis='both')
        fig.tight_layout()
        plt.show()
    
    def plot_specific_nodes(self, nodes=1):
        if isinstance(nodes, int):
            nodes = [nodes]
        
        # also include the nodes' neighbours
        g = recreate_graph(self.direc)
        neighbours = [edge for edge in g.edges(data='correlation') if nodes[0] in edge]
        neighbours = [i[0] if i[1]==nodes[0] else i[1] for i in neighbours]
        print(neighbours)
        nodes.extend(neighbours)
        
        if not self.summary_stats_dict:
            self.calculate_summary_statistics()
        
        for key, item in self.summary_stats_dict.items():
            self.summary_stats_dict[key] = item.T
        
        time_list = np.arange(self.summary_stats_dict['mean'].shape[-1])
        
        fig, ax = plt.subplots()
        for count, node in enumerate(nodes):
            ax.plot(time_list, self.summary_stats_dict['mean'][node], label=f'{node} {"+" if (node == nodes[0] or g.edges[node,nodes[0]]["correlation"] > 0) else "-"}')
        ax.grid(linestyle='--', axis='both')
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close()
    
    def plot_specific_nodes2(self, nodes=1):
        nodes = [nodes]
        
        # also include the nodes' neighbours
        g = recreate_graph(self.direc)
        neighbours = [edge for edge in g.edges(data='correlation') if nodes[0] in edge]
        neighbours = [i[0] if i[1]==nodes[0] else i[1] for i in neighbours]
        for u,v in itertools.combinations(neighbours, 2):
            if g.has_edge(u,v):
                print(f'{u} {v}: {g.edges[u,v]["correlation"]}')
        #nodes.extend(neighbours)
        
        asymp_time, asymp_median = get_asymp_info(self.direc)
        for n in nodes:
            print(f'Node {n}: asympt time {asymp_time[n]}')
        
        if not self.summary_stats_dict:
            self.calculate_summary_statistics()
        
            for key, item in self.summary_stats_dict.items():
                self.summary_stats_dict[key] = item.T
            
            self.summary_stats_dict['mean'] = self.summary_stats_dict['mean'][:, :]
            self.summary_stats_dict['stddev'] = self.summary_stats_dict['stddev'][:, :]
        time_list = np.arange(self.summary_stats_dict['mean'].shape[-1])
        
        fig, ax = plt.subplots()
        ally_count = 1
        opponent_count = 1
        for node in nodes:
            if node == nodes[0]:
                ax.plot(time_list, self.summary_stats_dict['mean'][node], label=f'Agent 1')
            elif g.edges[node,nodes[0]]["correlation"] > 0:
                ax.plot(time_list, self.summary_stats_dict['mean'][node], label=f'Ally {ally_count}')
                ally_count += 1
            else:
                ax.plot(time_list, self.summary_stats_dict['mean'][node], label=f'Opponent {opponent_count}')
                opponent_count += 1
        ax.grid(linestyle='--', axis='both')
        ax.set_xlabel(r't')
        ax.set_ylabel(r'$\langle \theta \rangle$')
        #ax.legend()
        fig.tight_layout()
        
        ally_count = 1
        opponent_count = 1
        fig, ax = plt.subplots()
        for count, node in enumerate(nodes):
            if node == nodes[0]:
                ax.plot(time_list, self.summary_stats_dict['stddev'][node], label=f'Agent 1')
            elif g.edges[node,nodes[0]]["correlation"] > 0:
                ax.plot(time_list, self.summary_stats_dict['stddev'][node], label=f'Ally {ally_count}')
                ally_count += 1
            else:
                ax.plot(time_list, self.summary_stats_dict['stddev'][node], label=f'Opponent {opponent_count}')
                opponent_count += 1
        ax.grid(linestyle='--', axis='both')
        ax.set_xlabel(r't')
        ax.set_ylabel(r'$\sqrt{\langle(\theta - \langle \theta \rangle)^2 \rangle}$')
        #ax.legend()
        fig.tight_layout()
        
        plt.show()
        plt.close()
        
    
    def check_prior_at_time(self, node=77, t=1000):
        #nodes = [70, 1, 22, 65]
        #nodes = [65, 3, 30, 40, 70]
        #nodes = [77, 12, 48, 60]
        nodes = [node]
        g = recreate_graph(self.direc)
        neighbours = [edge for edge in g.edges(data='correlation') if nodes[0] in edge]
        neighbours = [i[0] if i[1]==nodes[0] else i[1] for i in neighbours]
        nodes.extend(neighbours)
        
        priors = [extract_prior(self.direc, n, t) for n in nodes]
        bias_list = np.linspace(0,1,21)
        
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        line_list = []
        for n in range(len(nodes)):
            line_list.append(ax.plot(np.linspace(0,1,21), priors[n], label=f'{nodes[n]} {"+" if (n == 0 or g.edges[nodes[n],node]["correlation"] > 0) else "-"}')[0])
        ax.legend()
        ax.grid(linestyle='--', axis='both')
        ax.set_title(f't={t}')
        #fig.tight_layout()
        
        
        class Index(object):
            ind = 0

            def call_next(self2, event):
                self2.ind += 1
                self2.update_data()
                
            def call_prev(self2, event):
                self2.ind -= 1
                self2.update_data()

            def call_slider(self2, event):
                self2.ind = int(event)
                self2.update_data()
                
            def update_data(self2):
                priors = [extract_prior(self.direc, n, self2.ind) for n in nodes]
                for count, line in enumerate(line_list):
                    line.set_ydata(priors[count])
                ax.set(title=f't={self2.ind}')
                plt.draw()
                
        
        callback = Index()
        callback.ind = t
        ax_next = plt.axes([0.1, 0.07, 0.2, 0.05])  # left, top, width height
        next_button = Button(ax_next, 'Next')
        next_button.on_clicked(callback.call_next)
        
        ax_prev = plt.axes([0.4, 0.07, 0.2, 0.05])  # left, top, width height
        prev_button = Button(ax_prev, 'Prev')
        prev_button.on_clicked(callback.call_prev)
        
        plt.show()
              
    def check_enemy_agreement(self):
        if not self.summary_stats_dict:
            self.calculate_summary_statistics()
            
        g = recreate_graph(self.direc)
        mode_list = self.summary_stats_dict['mode'][-1]
        
        for u,v,correlation in g.edges(data='correlation'):
            if correlation < 0 and np.isclose(mode_list[u], mode_list[v]):
                print(f'{u}, {v} yeet')
                print(f'Degree of {u}: {g.degree(u)}')
                print(f'Degree of {v}: {g.degree(v)}')
                print(f'Mode of {u} and {v}: {mode_list[u]}')
                
                prior1 = extract_prior(self.direc, u)
                prior2 = extract_prior(self.direc, v)
                a = np.max(np.abs(prior1-prior2))
                b = np.max(np.maximum(prior1,prior2))
                print(f'Belief diff: {a/b}')
                
                common_list = [i for i in g.neighbors(u) if i in g.neighbors(v)]
                if common_list:
                    print(f'Common neighbours: {common_list}')
                    
                    for n in common_list:
                        print(f'Mode of {n}: {mode_list[n]}')
                
    def get_least_confident_node(self):
        if not self.summary_stats_dict:
            self.calculate_summary_statistics()
            
        g = recreate_graph(self.direc)
        # max_list = np.argmax(self.summary_stats_dict['stddev'], axis=1)
        max_list = np.argsort(self.summary_stats_dict['stddev'], axis=1)
        print(max_list[-1, -10:])  # time, node
         
    def check_asymp_time(self):
        asymp_time, asymp_median = get_asymp_info(self.direc)
        correct_asymp = []
        wrong_asymp = []
        
        for i in range(len(asymp_time)):
            if np.isclose(asymp_median[i], 0.6):
                correct_asymp.append(asymp_time[i])
            else:
                wrong_asymp.append(asymp_time[i])
                
        df = pd.DataFrame({'Correct': correct_asymp})
        df2 = pd.DataFrame({'Wrong': wrong_asymp})
        print(df.describe())
        print('')
        print(df2.describe())
        
        fig, ax = plt.subplots()
        ax.hist(correct_asymp)
        ax.grid(linestyle='--', axis='both')
        ax.set_title('Correct')
        fig.tight_layout()
        
        fig, ax = plt.subplots()
        ax.hist(wrong_asymp)
        ax.grid(linestyle='--', axis='both')
        ax.set_title('Wrong')
        fig.tight_layout()
        
        plt.show()

    def get_nodes_no_asymp(self):
        asymp_time, asymp_median = get_asymp_info(self.direc)
        
        has_asymp = asymp_time < (max(asymp_time) - 100)
        num_no_asymp = has_asymp.size - np.count_nonzero(has_asymp)
        print(f'Num no asymp: {num_no_asymp}')
        
        num_correct = np.count_nonzero(np.isclose(asymp_median[has_asymp], 0.6))
        print(f'Num correct: {num_correct}')
        
        g = recreate_graph(self.direc)
        for count, asymp in enumerate(has_asymp):
            num_allies = 0
            if not asymp:
                for u,v, correlation in g.edges(count, data='correlation'):
                    if correlation > 0:
                        num_allies += 1
                if num_allies == 0:
                    print(count)
                    print(asymp)
        
        return num_no_asymp

    def get_belief_separation(self,t=8000):
        g = recreate_graph(self.direc, t)
        max_separation = 0
        
        for n, belief in g.nodes(data='prior'):
            truth = belief > 0.01 * np.max(belief)
            first_true = np.argmax(truth)
            last_true = len(truth) - np.argmax(truth[::-1]) - 1
            
            separation = last_true - first_true
            if separation > max_separation:
                max_separation = separation
                
            if first_true == 16:
                print('Yeeee')
            elif first_true == 8:
                print('Yus')
                
                
        print(max_separation * 0.05)

def main():
    time1 = timeit.default_timer()

    plt.rcParams.update({'font.size': 15})
    analyser = DataAnalyser(f'output/temp')
    analyser.produce_plots()
    plt.rcParams.update({'font.size': 15})
    # analyser.find_convergence()
    # analyser.graph_convergence()
    # analyser.calculate_true_bias_estimates()

    # analyser.save_priors_into_gif()

    time2 = timeit.default_timer()
    print(f'Time taken: {(time2 - time1):.2f} seconds')
    

def plot_degree_distribution():
    m=3
    G = nx.barabasi_albert_graph(100, m)
    
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    fig, ax = plt.subplots()
    ax.loglog(degrees[m:], degree_freq[m:], 'o-') 
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(f'm={m}')
    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 17})
    
    # DataAnalyser('output/ba3').plot_specific_nodes2(70)
    # DataAnalyser('output/ba3').get_nodes_no_asymp()
    # DataAnalyser('output/ba3').check_prior_at_time(node=65, t=2700)
    # DataAnalyser('biased_results/output/default').produce_plots()
    # DataAnalyser('output/default').find_belief_convergence(True)
    # DataAnalyser('output/oscillation').find_belief_oscillation()
    # DataAnalyser('output/test_sus2').plot_specific_nodes(37)
    DataAnalyser('output/test_sus2').plot_specific_nodes(67)
    # DataAnalyser('output/test_sus2').get_least_confident_node()
    

"""
94
check is all no asymp has at least one ally.
"""   
