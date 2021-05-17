import numpy as np
import scipy as sp
import networkx as nx
import itertools
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator  # to set plot ticks to integers
from matplotlib.collections import LineCollection
from read_graph import extract_basic_info, remove_files, extract_prior, recreate_graph, extract_true_coin_bias
import timeit
from PIL import Image
import glob
import natsort as ns
from scipy import stats 

import sys
sys.path.append("..") # Adds higher directory to python modules path. (allows us to import parseargs)
from parseargs import parser_handle_ensemble
import argparse

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

    def __init__(self, direc='output/test'):
        self.direc = direc
        self.file_name_prepend = ''
        self.summary_stats_list = []

    def calculate_summary_statistics(self):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)

        mean_list = np.zeros((len(time_list), number_of_nodes))
        mode_list = np.zeros((len(time_list), number_of_nodes))
        stddev_list = np.zeros((len(time_list), number_of_nodes))
        param_list = [['mean', mean_list], ['mode', mode_list], ['stddev', stddev_list]]  # for looping the parameters

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

        self.summary_stats_list = param_list
        return param_list

    def produce_plots(self):
        number_of_nodes, time_list, bias_list = extract_basic_info(self.direc)
        num_iter = len(time_list)

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
                    ax.plot(bias_list, prior, label=str(i))
                else:
                    ax.plot(bias_list, prior)

            ax.set_xlabel('Bias')
            ax.set_xlabel(r'$\theta$')
            ax.set_ylabel('Probability')
            ax.grid(linestyle='--', axis='x')
            if t == 0:
                ax.set_title('Initial beliefs')
            else:
                ax.set_title('Iteration {}'.format(t))

            if number_of_nodes < 8:
                ax.legend()
            plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
            plt.tight_layout()
            fig.savefig(f'{self.direc}/prior/{self.file_name_prepend}t={t}.png')
            plt.close()

            if t == 0:
                fig, ax = plt.subplots()
                for i in range(number_of_nodes):
                    prior = extract_prior(self.direc, node=i, t=t)
                    ax.plot(bias_list, prior*sp.stats.binom.pmf(5, 10, bias_list) / np.sum(prior * sp.stats.binom.pmf(5, 10, bias_list)), label=str(i))

                ax.set_xlabel('Bias')
                ax.set_xlabel(r'$\theta$')
                ax.set_ylabel('Probability')
                ax.grid(linestyle='--', axis='x')
                plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
                plt.tight_layout()
                fig.savefig(f'{self.direc}/prior/{self.file_name_prepend}ideal.png')
                plt.close()

        # calculate mean, mode and std dev
        param_list = self.calculate_summary_statistics()

        for param_name, data in param_list:
            remove_files(f'{self.direc}/{param_name}', ends_with='.png')

        for t in time_list:
            # also plot histograms of the parameters at a certain time
            if t in chopped_time_list:
                for pair in param_list:
                    param_name = pair[0]
                    data = pair[1][t]

                    fig, ax = plt.subplots()
                    ax.hist(data, bins=np.arange(np.min(data)-0.025, np.max(data) + 0.05, 0.05))
                    ax.set_xlabel(param_name.capitalize())
                    ax.set_ylabel('Count')
                    ax.set_title('Iteration {}'.format(t))
                    ax.grid(linestyle='--', axis='y')
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # set the y ticks to integers only
                    plt.tight_layout()
                    fig.savefig('{}/{}/t={}'.format(self.direc, param_name, t))
                    plt.close()

        # time series - boxplot
        remove_files(f'{self.direc}/time_series', ends_with='.png')

        # time series - boxplot. boxplots only worth it for enough data points
        # TODO separate plots if time_list is too big
        if number_of_nodes > 10 and len(time_list) < 20:
            for param_name, data in param_list:
                fig, ax = plt.subplots()
                ax.boxplot(data, positions=time_list)  # TODO figure out how to show certain ticks only
                ax.set_xlabel('Iteration')
                ax.set_ylabel(param_name.capitalize())
                ax.grid(linestyle='--')
                plt.tight_layout()
                fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}box_{param_name}.png')
                plt.close()

        # time series - dots
        for param_name, data in param_list:
            fig, ax = plt.subplots()
            for i in time_list:
                x_value = np.ones(len(data[i])) * i
                ax.plot(x_value, data[i], 'bo')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param_name.capitalize())
            ax.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}dots_{param_name}.png')
            plt.close()

            # plot second half of the time, to 'zoom' in
            fig, ax = plt.subplots()
            midpoint = len(data) // 2
            halved_time_list = time_list[midpoint:]
            for i in halved_time_list:
                x_value = np.ones(len(data[i])) * i
                ax.plot(x_value, data[i], 'bo')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param_name.capitalize())
            ax.grid(linestyle='--')
            plt.tight_layout()
            fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}dots_chopped_{param_name}.png')
            plt.close()

        # draw the network. Above 20 nodes, don't do this because it gets too messy
        if 2 < number_of_nodes < 20:
            mean_list = param_list[0][1]
            g = recreate_graph(self.direc)
            edge_data = [x[2] for x in g.edges.data('correlation')]

            # edge_data = [x[2] for x in g.edges.data('correlation')]
            layout = nx.spring_layout(g)  # to keep the position of the nodes consistent between iterations

            remove_files(f'{self.direc}/network', ends_with='.png')
            for t in chopped_time_list:
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
        for i in range(len(param_list)):
            param_list[i][1] = np.transpose(param_list[i][1])

        # time series - lines
        # g = recreate_graph(self.direc)
        # for param_name, data in param_list:
        #     fig, ax = plt.subplots()
        #     big_time_list = []
        #     big_node_data_list = []
        #     color_list = []
        #     node_number = 0
        #     for node_data in data:
        #         big_time_list.append(time_list)
        #         big_node_data_list.append(node_data)
        #         color_list.append(len(list(g.neighbors(node_number))))
        #         node_number += 1
        #     plot = multiline(big_time_list, big_node_data_list, color_list, ax=ax, lw=1, cmap='viridis')
        #     cbar = fig.colorbar(plot)
        #     cbar.set_label('Node degree')
        #     ax.set_xlabel('Iteration')
        #     ax.set_ylabel(param_name.capitalize())
        #
        #     if param_name != 'stddev':
        #         start, end = ax.get_ylim()
        #         first = np.ceil(start*10) / 10
        #         last = np.ceil(end*10) / 10
        #         ax.set_yticks(np.arange(first, last, 0.1))
        #     ax.grid(linestyle='--')
        #     plt.tight_layout()
        #     fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}lines_{param_name}.png')
        #     plt.close()

        for param_name, data in param_list:
            fig, ax = plt.subplots()
            for node, node_data in enumerate(data):
                ax.plot(time_list, node_data, label=str(node))
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
            fig.savefig(f'{self.direc}/time_series/{self.file_name_prepend}lines_{param_name}.png')
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
        if not self.summary_stats_list:
            self.calculate_summary_statistics()

        # exclude first data
        mean_list = self.summary_stats_list[0][1][1:]
        mode_list = self.summary_stats_list[1][1][1:]
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

        # for mean_node_list in mean_list:
        #     num_correct = 0
        #     for mean in mean_node_list:
        #         if true_bias - correct_width < mean < true_bias + correct_width:
        #             num_correct += 1
        #
        #     correct_mean_list.append(num_correct)
        #
        # for mode_node_list in mode_list:
        #     num_correct = 0
        #     for mode in mode_node_list:
        #         if true_bias - correct_width < mode < true_bias + correct_width:
        #             num_correct += 1
        #
        #     correct_mode_list.append(num_correct)


        # fig, ax = plt.subplots()
        #
        # ax.plot(correct_mean_list, label='Mean')
        # ax.plot(correct_mode_list, label='Mode')
        # ax.legend()
        # plt.show()

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

def main(args):

    num_runs = args.runs if not args.single else 1

    fname_pref = 'output/test' + ('_' if not args.single else '')

    print("will read from", fname_pref, "for", num_runs, "runs")

    for i in range(num_runs):
        time1 = timeit.default_timer()

        fname = fname_pref
        if not args.single:
            fname += str(i)
        print('reading from', fname)
        plt.rcParams.update({'font.size': 15})
        analyser = DataAnalyser(fname)
        analyser.produce_plots()
        plt.rcParams.update({'font.size': 15})
        # analyser.find_convergence()
        # analyser.graph_convergence()
        # analyser.calculate_true_bias_estimates()

        # analyser.save_priors_into_gif()

        time2 = timeit.default_timer()
        print(f'Time taken: {(time2 - time1):.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Opinion dynamics simulation analysis, Yutong Bu 2021')

    parser_handle_ensemble(parser)
    
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("number of ensemble simulation runs must be >=1")

    main(args)
