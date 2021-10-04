import simulation, analyse
import numpy as np
import networkx as nx
from read_graph import recreate_graph
from generate_graph import unbalanced_triangle, find_frac_of_unbalanced_cycles
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from copy import deepcopy
import sys, os, itertools
import pandas as pd
from scipy.optimize import curve_fit


"""
Used whenever I need to use something from simulation.py and analyse.py at the same time.
Most functions here are obselete, now that simulation_numba.py exists.
They're only used when I need a plot of mean vs time, like Figure 2, of one particular simulation

If you need to do that, the way to do it is like this:

save_folder = 'output/default'
g = nx.barabasi_albert_graph(100, 3, seed=100)
g = simulation.randomise_prior(g)  # or some other way of filling in g.nodes[node]['prior']
simulator = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, output_direc=save_folder, coin_obs_file=None, bias_len=21, asymp_time_threshold=100)
simulator.do_simulation(num_iter=10000, blend_method=0)
analyser = analyse.DataAnalyser(save_folder).produce_plots()

If you want to provide the coin tosses, either fill in coin_obs_file=reference_folder to copy the coin tosses of a previous simulation
Or fill in simulator.heads_list = heads_list and
simulator.tosses_list = np.ones(heads_list.shape, dtype=int)
before calling simulator.do_simulation
"""


class HiddenPrints:
    """
    Context manager to ignore any print statements. Usage:
    with HiddenPrints():
        print('Yo')  # this won't actually print anything
    print('YOYO')  # Now this will print
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def compare_triangles(method=0):
    # compare between four different triads
    num_iter = 10000
    repetitions = 100
    repetition10th = 10 ** (int(np.log10(repetitions)) - 1)
    
    # NOTE: does not actually support node numbers != 3 properly. For now
    num_nodes = 3
    num_edges = num_nodes * (num_nodes - 1) // 2

    possible_edge_weight_combo = list(itertools.combinations_with_replacement([1,-1], num_edges))
    possible_edge_combo = list(itertools.combinations(list(range(num_nodes)), 2))
    tosses_list = np.ones(num_iter, dtype=int) * 1
    
    category_dict = {'Triangle': [], 'Time': [], 'All agree': [], 'Num right': [], 'Num almost right':[], 'First right': [], 'Width': []}
    
    for k in range(repetitions):
        if k % repetition10th == 0:
            print(f'Iteration {k}')
        g1 = nx.complete_graph(num_nodes)
        g1 = simulation.randomise_prior(g1)
        
        heads_list = np.random.binomial(1, 0.5, num_iter)
        
        for i in range(len(possible_edge_weight_combo)):
            g2 = g1.copy()
            for j in range(num_edges):
                g2.edges[possible_edge_combo[j]]['correlation'] = possible_edge_weight_combo[i][j]
                
            simulator = simulation.Simulation(g2.copy(), true_bias=0.5, tosses_per_iteration=1, bias_len=21)
            simulator.heads_list = heads_list
            simulator.tosses_list = tosses_list
            
            t = simulator.do_simulation_without_print_and_save(num_iter=num_iter, blend_method=0)
            all_agree = int(simulator.do_all_nodes_agree)
            first_right = int(simulator.is_first_node_correct)
            num_right = simulator.number_of_correct_nodes(opinion_range=(0.49,0.51), criteria='mode')
            num_almost_right = simulator.number_of_correct_nodes(opinion_range=(0.44,0.56), criteria='mode')
            width = np.abs(simulator.get_width).max()
            
            category_dict['Triangle'].append(i)
            category_dict['Time'].append(t)
            category_dict['All agree'].append(all_agree)
            category_dict['Num right'].append(num_right)
            category_dict['Num almost right'].append(num_almost_right)
            category_dict['First right'].append(first_right)
            category_dict['Width'].append(width)
                
    def struc_balance(combo):
        num_negative = len([i for i in combo if i < 0])
        if num_negative == 1:
            return 'Unbalanced'
        elif num_negative % 2 == 0:
            return 'Strong'
        else:
            return 'Weak'
            
    triangle_ref = {}
    for i in range(len(possible_edge_weight_combo)):
        triangle_ref[i] = [combo for combo in possible_edge_weight_combo[i]]
        triangle_ref[i].append(struc_balance(possible_edge_weight_combo[i]))
    
    # df = pd.DataFrame(results)
    df2 = pd.DataFrame(category_dict)
    df3 = pd.DataFrame(triangle_ref)
    with pd.ExcelWriter("results.xlsx") as writer:
        # df.to_excel(writer, index=False, sheet_name='Results')
        df2.to_excel(writer, index=False, sheet_name='By category')
        df3.to_excel(writer, index=False, sheet_name='Legend')
            
            
def compare_methods():
    # compare between four different triads and three different methods
    method_list = [0,1,2]
    for method in method_list:
        output_direc = f'output/compare_triangles{method}'
    os.makedirs(output_direc, exist_ok=True)
    num_iter = 10000
    repetitions = 1000
    repetition10th = 10 ** (int(np.log10(repetitions)) - 1)
    
    # NOTE: does not actually support node numbers != 3 properly. For now
    num_nodes = 3
    num_edges = num_nodes * (num_nodes - 1) // 2

    possible_edge_weight_combo = list(itertools.combinations_with_replacement([1,-1], num_edges))
    possible_edge_combo = list(itertools.combinations(list(range(num_nodes)), 2))
    tosses_list = np.ones(num_iter) * 10
    
    results_list = np.zeros((len(possible_edge_weight_combo), repetitions), dtype=int)
    
    # results = {}
    category_dict = {'Method': [], 'Triangle': [], 'Time': [], 'All agree': [], 'Num right': [], 'Num almost right':[], 'First right': [], 'Width': []}
    # for i in range(len(possible_edge_weight_combo)):
    #     for method in method_list:
    #         results[f'method{method}tri{i}'] = []
    #         results[f'{method}{i}agree'] = []
    #         results[f'{method}{i}num_right'] = []
    #         results[f'{method}{i}num_almost_right'] = []
    
    for k in range(repetitions):
        if k % repetition10th == 0:
            print(f'Iteration {k}')
        g1 = nx.complete_graph(num_nodes)
        g1 = simulation.randomise_prior(g1)
        
        heads_list = np.random.binomial(10, 0.5, num_iter)
        
        for i in range(len(possible_edge_weight_combo)):
            g2 = g1.copy()
            for j in range(num_edges):
                g2.edges[possible_edge_combo[j]]['correlation'] = possible_edge_weight_combo[i][j]
                
            for method in method_list:
                simulator = simulation.Simulation(g2.copy(), true_bias=0.5, tosses_per_iteration=10, bias_len=21)
                simulator.heads_list = heads_list
                simulator.tosses_list = tosses_list
                
                t = simulator.do_simulation_without_print_and_save(num_iter=num_iter, blend_method=method)
                all_agree = int(simulator.do_all_nodes_agree)
                num_right = simulator.number_of_correct_nodes(opinion_range=(0.49,0.51), criteria='mode')
                num_almost_right = simulator.number_of_correct_nodes(opinion_range=(0.44,0.56), criteria='mode')
                first_right = int(simulator.is_first_node_correct)
                width = np.abs(simulator.get_width).max()
                
                # results[f'method{method}tri{i}'].append(t)
                # results[f'{method}{i}agree'].append(all_agree)
                # results[f'{method}{i}num_right'].append(num_right)
                # results[f'{method}{i}num_almost_right'].append(num_almost_right)
                
                category_dict['Method'].append(method)
                category_dict['Triangle'].append(i)
                category_dict['Time'].append(t)
                category_dict['All agree'].append(all_agree)
                category_dict['Num right'].append(num_right)
                category_dict['Num almost right'].append(num_almost_right)
                category_dict['First right'].append(first_right)
                category_dict['Width'].append(width)
                
    def struc_balance(combo):
        num_negative = len([i for i in combo if i < 0])
        if num_negative == 1:
            return 'Unbalanced'
        elif num_negative % 2 == 0:
            return 'Strong'
        else:
            return 'Weak'
            
    triangle_ref = {}
    for i in range(len(possible_edge_weight_combo)):
        triangle_ref[i] = [combo for combo in possible_edge_weight_combo[i]]
        triangle_ref[i].append(struc_balance(possible_edge_weight_combo[i]))
    
    # df = pd.DataFrame(results)
    df2 = pd.DataFrame(category_dict)
    df3 = pd.DataFrame(triangle_ref)
    with pd.ExcelWriter("results.xlsx") as writer:
        # df.to_excel(writer, index=False, sheet_name='Results')
        df2.to_excel(writer, index=False, sheet_name='By category')
        df3.to_excel(writer, index=False, sheet_name='Legend')

            
def read_results(filename='results.xlsx'):
    df = pd.read_excel(filename, sheet_name='By category')
    fig, axes = plt.subplots(2,3)
    ax_list = axes.flatten()[:-1]
    label_list = ['Time', 'All agree', 'Num right', 'Num almost right', 'First right', 'Width']
    marker_list = 'oxD'
    for method in range(df['Method'].max() + 1):
        mean_list = [[] for i in range(len(label_list))]
        data_range = []
        for triangle in range(4):
            data = df[(df['Method'] == method) & (df['Triangle'] == triangle)]
            nonzero_data = data[data['Time'] > 0]
            if nonzero_data.shape[0] > 0:
                for count, label in enumerate(label_list):
                    mean_list[count].append(nonzero_data[label].mean())
                data_range.append(triangle)
        
        for count, ax in enumerate(ax_list):
            ax.plot(data_range, mean_list[count], label=str(method), lw=0.5, marker=marker_list[method])
    
    for count, ax in enumerate(ax_list):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Triangle')
        ax.set_ylabel(label_list[count])
        ax.legend(title='Method')
        ax.grid(linestyle='--', axis='both')
        
    fig.tight_layout()
        
    plt.show()
    plt.close()
    

def read_results_nice_plots(filename='results.xlsx'):
    df = pd.read_excel(filename, sheet_name='By category')
    label_list = ['Time', 'All agree', 'Num right', 'Num almost right', 'First right', 'Width']
    subplot_list = [plt.subplots(figsize=(5,4)) for i in range(len(label_list))]
    fig_list = [i[0] for i in subplot_list]
    ax_list = [i[1] for i in subplot_list]
    
    mean_list = [[] for i in range(len(label_list))]
    data_range = []
    for triangle in range(4):
        data = df[df['Triangle'] == triangle]
        # nonzero_data = data[data['Time'] > 0]
        nonzero_data = data[data['Time'] > -1]
        if nonzero_data.shape[0] > 0:
            for count, label in enumerate(label_list):
                mean_list[count].append(nonzero_data[label].mean())
            data_range.append(triangle)
    
    for count, ax in enumerate(ax_list):
        ax.plot(data_range, mean_list[count], lw=2,  ms=10)
    
    for count, ax in enumerate(ax_list):
        x_labels = ['a', r'$G_0$', r'$G_1$', r'$G_2$', r'$G_3$']
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Triangle')
        ax.set_ylabel(label_list[count])
        ax.grid(linestyle='--', axis='both')
        ax.set_xticklabels(x_labels)
        
    for count, fig in enumerate(fig_list):
        fig.tight_layout()
        fig.savefig(f'figs/triangles/{label_list[count]}.png')
        
    plt.close()
    # 
    # group_by = df[df['Time'] > 0].groupby(['Triangle'])
    # print(group_by.count())
    # print('')
    # print(group_by.mean())


def read_results_triangles():
    filename = 'results.xlsx'
    df = pd.read_excel(filename, sheet_name='By category')
    print(df['Right-wrong'].describe())
    print(df[df['Right-wrong'] > 0]['Right-wrong'].count())
    print(df[df['Right-wrong'] < 0]['Right-wrong'].count())
    
    fig, ax = plt.subplots()
    ax.hist(df['Right-wrong'], bins='auto')
    ax.set_xlabel(r'$t_{\rm a}^{\rm right}-t_{\rm a}^{\rm wrong}$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    fig.tight_layout()
    plt.show()
    
            

def read_results_nice_plots_for_methods(filename='results.xlsx'):
    df = pd.read_excel(filename, sheet_name='By category')
    label_list = ['Time', 'All agree', 'Num right', 'Num almost right', 'First right', 'Width']
    subplot_list = [plt.subplots(figsize=(5,4)) for i in range(len(label_list))]
    fig_list = [i[0] for i in subplot_list]
    ax_list = [i[1] for i in subplot_list]
    
    marker_list = 'oxD'
    for method in range(2):
        mean_list = [[] for i in range(len(label_list))]
        data_range = []
        for triangle in range(4):
            data = df[(df['Method'] == method) & (df['Triangle'] == triangle)]
            nonzero_data = data[data['Time'] > 0]
            if nonzero_data.shape[0] > 0:
                for count, label in enumerate(label_list):
                    mean_list[count].append(nonzero_data[label].mean())
                data_range.append(triangle)
        
        for count, ax in enumerate(ax_list):
            ax.plot(data_range, mean_list[count], label=str(method), lw=2, marker=marker_list[method], ms=10)
    
    for count, ax in enumerate(ax_list):
        x_labels = ['a', r'$G_0$', r'$G_1$', r'$G_2$', r'$G_3$']
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Triangle')
        ax.set_ylabel(label_list[count])
        ax.legend(title='Method')
        ax.grid(linestyle='--', axis='both')
        ax.set_xticklabels(x_labels)
        
    for count, fig in enumerate(fig_list):
        fig.tight_layout()
        fig.savefig(f'figs/{label_list[count]}.png')
        
    plt.close()
    
    group_by = df[df['Time'] > 0].groupby(['Triangle', 'Method'])
    print(group_by.count())
    print('')
    print(group_by.mean())


def two_nodes_reach_wrong_conclusion_first():
    bias_len = 21
    bias_list = np.linspace(0,1,bias_len)
    results_dict = {'Diff': []}
    
    max_time = 100000
    repetitions = 10000
    repetition10th = repetitions // 100
    a = 0  # 193
    b = 0  # 0
    
    for i in range(repetitions):
        if i % repetition10th == 0:
            print(i)
        g1 = nx.complete_graph(2)
        g1 = simulation.randomise_prior(g1)
        g1.edges[0,1]['correlation'] = -1
        
        simulator = simulation.Simulation(g1, true_bias=0.6, tosses_per_iteration=1, bias_len=bias_len)
        simulator.do_simulation_without_print_and_save(num_iter=max_time, blend_method=0)
        if simulator.do_all_nodes_agree:
            continue
        time_list = simulator.get_asymp_time
        apparent_bias = np.argmax(simulator.apparent_true_bias) * 0.05
        if time_list[1][2] - 0.01 < time_list[1][1] < time_list[1][2] + 0.01:
            a += 1
            correct_node = 1
        elif (time_list[0][2] - 0.01 < time_list[0][1] < time_list[0][2] + 0.01) or (time_list[0][2] - 0.01 < time_list[1][1] < time_list[0][2] + 0.01):
            correct_node = 0
        else:
            print('b')
            b += 1
            correct_node = 0
        time_diff = time_list[1-correct_node][0] - time_list[correct_node][0]
        results_dict['Diff'].append(time_diff)
        
    df = pd.DataFrame(results_dict)
    df2 = pd.DataFrame({'Num trials': [repetitions]})
    with pd.ExcelWriter("results_4.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        df2.to_excel(writer, index=False, sheet_name='Number of trials')
    
    two_nodes_read_results("results_4.xlsx")
   

def two_nodes_read_results(excel_file='results.xlsx'):
    
    df = pd.read_excel(excel_file)
    # print(df.describe())
    # print(df[df['Diff'] > 0].count())
    # print(df[df['Diff'] < 0].count())
    # print(df.quantile((0.05,0.95)))
     
    # fig, ax = plt.subplots()
    # ax.hist(df['Diff'], bins='auto', range=(-19, 10))
    # ax.set_xlabel(r'$t_a^{\rm right} - t_a^{\rm wrong}$')
    # ax.set_ylabel('Count')
    # ax.grid(linestyle='--', axis='both')
    # fig.tight_layout()
    # #plt.show()
    
    fig, ax = plt.subplots()
    
    ax.hist(df['Diff'], bins='auto')
    ax.set_xlabel(r'$t_a^{\rm right} - t_a^{\rm wrong}$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    fig.tight_layout()
    
    axin = inset_axes(ax, width='50%', height='40%', loc=1)
    axin.hist(df['Diff'], bins=np.arange(-18.5,10.5, 1))  # do integer bins instead
    axin.grid(linestyle='--', axis='both')
    
    plt.show()
    
    # hist, bins = np.histogram(df['Diff'], bins='auto')
    # arg_max = np.argmax(hist)
    # 
    # bins_middle = []
    # for i in range(len(bins) - 1):
    #     bins_middle.append((bins[i] + bins[i+1]) / 2)
    #     
    # def power_law(x, a, b, c, d):
    #     return a + c * np.exp(b * (x-d))
    #     
    # stop_index = 0
    # for i in range(len(bins_middle)):
    #     if bins_middle[i] < 4000 < bins_middle[i+1]:
    #         stop_index = i
    #         break
    #     
    # params, cov = curve_fit(power_law, bins_middle[arg_max:stop_index], hist[arg_max:stop_index], bounds=(-np.inf, (np.inf, 0, np.inf, np.inf)), p0=(1,-1,1,1))
    # print(cov)
    # 
    # ax.plot(bins_middle[arg_max:stop_index], power_law(bins_middle[arg_max:stop_index], *params))
    # plt.show()


def two_nodes_test():
    folder1 = 'output/temp'

    bias_len = 21
    bias_list = np.linspace(0,1,bias_len)
    g = nx.complete_graph(2)
    for edge in g.edges():
        g.edges[edge]['correlation'] = -1
    # g.nodes[1]['prior'] = simulation.gaussian_stddev(bias_list, mean=0.25, stddev=0.1)
    # g.nodes[0]['prior'] = simulation.gaussian_stddev(bias_list, mean=0.75, stddev=0.1)
    g.nodes[1]['prior'] = simulation.gaussian_stddev(bias_list, mean=0.65, stddev=10000000)
    g.nodes[0]['prior'] = simulation.gaussian_stddev(bias_list, mean=0.55, stddev=10000000)
    
    simulator = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=bias_len, output_direc=folder1)
    simulator.do_simulation(num_iter=10000, blend_method=0)
    analyser = analyse.DataAnalyser(folder1)
    #analyser.find_belief_convergence()
    analyser.produce_plots()
    print(simulator.get_asymp_time)
    print(simulator.get_proper_asymp_time)
    
    # g = recreate_graph('output/default', 0)
    # for edge in g.edges():
    #     g.edges[edge]['correlation'] = -1
    # simulator2 = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=bias_len, output_direc='output/default2')
    # simulator2.heads_list = simulator.heads_list
    # simulator2.tosses_list = simulator.tosses_list
    # simulator2.do_simulation(num_iter=10000, blend_method=0)
    # analyser = analyse.DataAnalyser('output/default2')
    # #analyser.find_belief_convergence()
    # #analyser.produce_plots()
    # 
    # print(simulator2.get_asymp_time)
    # print(simulator2.get_proper_asymp_time)
    
    # yeet_counter = 0
    # same_counter = 0
    # 
    # for _ in range(100):
    #     bias_len = 21
    #     bias_list = np.linspace(0,1,bias_len)
    #     g = nx.complete_graph(2)
    #     for edge in g.edges():
    #         g.edges[edge]['correlation'] = -1
    #     # g.nodes[1]['prior'] = simulation.gaussian_stddev(bias_list, mean=0.25, stddev=0.1)
    #     # g.nodes[0]['prior'] = simulation.gaussian_stddev(bias_list, mean=0.75, stddev=0.1)
    #     g.nodes[1]['prior'] = simulation.gaussian_stddev(bias_list, mean=np.random.random(), stddev=int(1e7))
    #     g.nodes[0]['prior'] = simulation.gaussian_stddev(bias_list, mean=np.random.random(), stddev=int(1e7))
    #     
    #     if np.all(g.nodes[0]['prior'] == g.nodes[1]['prior']) or np.all(g.nodes[0]['prior'][0] == g.nodes[0]['prior']) or np.all(g.nodes[1]['prior'][0] == g.nodes[1]['prior']):
    #         continue
    #     
    #     simulator = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=bias_len, output_direc=folder1)
    #     with HiddenPrints():
    #         simulator.do_simulation(num_iter=10000, blend_method=0)
    #     asymp_time = simulator.get_asymp_time
    #     
    #     if not (asymp_time[0][1] == 0.6 and asymp_time[1][1] != 0.6):
    #         print(asymp_time)
    #         yeet_counter += 1
    #         
    #     if asymp_time[0][1] == asymp_time[1][1]:
    #         print('What the hey')
    #         same_counter += 1
    # 
    # print(yeet_counter)
    # print(same_counter)
    

def recreate_simulation():
    folder = 'output/temp'
    g = recreate_graph(folder, 0)
    temp = g.nodes[0]['prior']
    g.nodes[0]['prior'] = g.nodes[1]['prior']
    g.nodes[1]['prior'] = temp
    simulator = simulation.Simulation(g, true_bias=0.5, tosses_per_iteration=1, bias_len=21, coin_obs_file=folder, output_direc=folder)
    simulator.do_simulation(num_iter=10000, blend_method=0)
    print(len(simulator.heads_list))
    print(simulator.get_proper_asymp_time)
    analyse.DataAnalyser(folder).produce_plots()


def test():
    # # used for unbalanced triad tests
    bias_len = 21
    # folder = 'output/temp'
    # g = nx.complete_graph(3)
    # g.edges[0,1]['correlation'] = -1
    # g.edges[2,1]['correlation'] = 1
    # g.edges[0,2]['correlation'] = 1
    # g = simulation.randomise_prior(g, bias_len=bias_len, mean=(0.3, 0.9))
    # 
    # # g = recreate_graph('output/triad', 0)
    # 
    # simulator = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=bias_len, output_direc=folder, coin_obs_file='output/triad')
    # with HiddenPrints():
    #     simulator.do_simulation(num_iter=2000, blend_method=0, learning_rate=0.25)
    # print(simulator.is_first_node_correct)
    # print(simulator.get_asymp_time)
    # print(simulator.get_proper_asymp_time)
    # analyse.DataAnalyser(folder).produce_plots()
    
    folder = 'output/default'
    coin_obs_file = 'output/pair_ally'
    # g = nx.barabasi_albert_graph(100,10,seed=100)
    g = nx.barabasi_albert_graph(100, 3, seed=100)
    g = simulation.randomise_prior(g)
    
    for edge in g.edges():
        g.edges[edge]['correlation'] = 1
    
    # g = recreate_graph(folder, 0)
    
    simulator = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=21, output_direc=folder, coin_obs_file=coin_obs_file)
    simulator.do_simulation(num_iter=10000, blend_method=0, learning_rate=0.25)
    analyser = analyse.DataAnalyser(folder)
    analyser.find_belief_convergence(True)
    analyser.produce_plots()
    
    folder2 = folder + '2'
    g = recreate_graph(folder, 0)
    for edge in g.edges():
        g.edges[edge]['correlation'] = -1
    simulator2 = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=bias_len, output_direc=folder2, coin_obs_file=coin_obs_file)
    simulator2.heads_list = simulator.heads_list
    simulator2.tosses_list = simulator.tosses_list
    simulator2.do_simulation(num_iter=10000, blend_method=0)
    # print(simulator2.get_proper_asymp_time)
    analyse.DataAnalyser(folder2).produce_plots()
    
    
    folder3 = folder + '3'
    g = recreate_graph(folder, 0)
    for edge in g.edges():
        if np.random.random() < 0.5:
            g.edges[edge]['correlation'] = -1
        else:
            g.edges[edge]['correlation'] = 1
    simulator3 = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=bias_len, output_direc=folder3, coin_obs_file=coin_obs_file)
    simulator3.heads_list = simulator.heads_list
    simulator3.tosses_list = simulator.tosses_list
    simulator3.do_simulation(num_iter=10000, blend_method=0)
    analyse.DataAnalyser(folder3).produce_plots()
    
    
    
    
def two_nodes_reach_wrong_conclusion_first_mu():
    # bias_len = 21
    # bias_list = np.linspace(0,1,bias_len)
    # mu_list = np.arange(0.0001, 0.0051, 0.0001)
    # results_dict = {}
    # for mu in mu_list:
    #     results_dict[mu] = []
    # 
    # max_time = 100000
    # repetitions = 1000
    # 
    # for mu in mu_list:
    #     print(mu)
    #     for i in range(repetitions):
    #         g1 = nx.complete_graph(2)
    #         g1 = simulation.randomise_prior(g1)
    #         g1.edges[0,1]['correlation'] = -1
    #         
    #         simulator = simulation.Simulation(g1, true_bias=0.5, tosses_per_iteration=1, bias_len=bias_len)
    #         simulator.do_simulation_without_print_and_save(num_iter=max_time, blend_method=0, learning_rate=mu)
    #         if simulator.do_all_nodes_agree:
    #             results_dict[mu].append(max_time)
    #             continue
    #         time_list = simulator.get_asymp_time
    #         apparent_bias = np.argmax(simulator.apparent_true_bias) * 0.05
    #         if time_list[1][2] - 0.01 < time_list[1][1] < time_list[1][2] + 0.01:
    #             correct_node = 1
    #         elif (time_list[0][2] - 0.01 < time_list[0][1] < time_list[0][2] + 0.01) or (time_list[0][2] - 0.01 < time_list[1][1] < time_list[0][2] + 0.01):
    #             correct_node = 0
    #         else:
    #             correct_node = 0
    #         time_diff = time_list[1-correct_node][0] - time_list[correct_node][0]
    #         results_dict[mu].append(time_diff)
    #     
    # df = pd.DataFrame(results_dict)
    # df2 = pd.DataFrame({'Num trials': [repetitions]})
    # with pd.ExcelWriter("results2.xlsx") as writer:
    #     df.to_excel(writer, index=False, sheet_name='Results')
    #     df2.to_excel(writer, index=False, sheet_name='Number of trials')
    # 
    # nonzero_data = df[df != max_time]
    # print(nonzero_data.describe())
    # print(nonzero_data.count())
    # 
    # fig, ax = plt.subplots()
    # ax.plot(mu_list, nonzero_data.count())
    # ax.set_xlabel(r'$\mu$')
    # ax.set_ylabel(r'Number of disagreements')
    # fig.tight_layout()
    # plt.show()

    bias_len = 21
    bias_list = np.linspace(0,1,bias_len)
    mu_list = np.arange(0.0001, 0.0051, 0.0001)
    results_dict = {}
    for mu in mu_list:
        results_dict[mu] = []
    num_agree_dict = {}
    for mu in mu_list:
        num_agree_dict[mu] = [0,0]  # [num_agree, total_sims]
    
    max_time = 100000
    repetitions = 1000
    
    for mu in mu_list:
        print(mu)
        i = 0
        while i < repetitions:
            g1 = nx.complete_graph(2)
            g1 = simulation.randomise_prior(g1)
            g1.edges[0,1]['correlation'] = -1
            
            simulator = simulation.Simulation(g1, true_bias=0.5, tosses_per_iteration=1, bias_len=bias_len)
            simulator.do_simulation_without_print_and_save(num_iter=max_time, blend_method=0, learning_rate=mu)
            if simulator.do_all_nodes_agree:
                num_agree_dict[mu][0] += 1
                num_agree_dict[mu][1] += 1
                continue
            time_list = simulator.get_asymp_time
            apparent_bias = np.argmax(simulator.apparent_true_bias) * 0.05
            if time_list[1][2] - 0.01 < time_list[1][1] < time_list[1][2] + 0.01:
                correct_node = 1
            elif (time_list[0][2] - 0.01 < time_list[0][1] < time_list[0][2] + 0.01) or (time_list[0][2] - 0.01 < time_list[1][1] < time_list[0][2] + 0.01):
                correct_node = 0
            else:
                correct_node = 0
            time_diff = time_list[1-correct_node][0] - time_list[correct_node][0]
            results_dict[mu].append(time_diff)
            num_agree_dict[mu][1] += 1
            i += 1
        
    df = pd.DataFrame(results_dict)
    df2 = pd.DataFrame(num_agree_dict)
    with pd.ExcelWriter("results2.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Right-wrong')
        df2.to_excel(writer, index=False, sheet_name='Agreements-trials')
    
    percent_disagree = [1 - df2[mu].iloc[0] / df2[mu].iloc[1] for mu in mu_list]
    fig, ax = plt.subplots()
    ax.plot(mu_list, percent_disagree)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$D$')
    ax.grid(linestyle='--', axis='both')
    fig.tight_layout()
    plt.show()
    
    median = df.median()
    first = df.quantile(0.25)
    third = df.quantile(0.75)
    
    fig, ax = plt.subplots()
    ax.errorbar(mu_list, median, yerr=(median-first, third-median))
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$t$')
    ax.grid(linestyle='--', axis='both')
    fig.tight_layout()
    plt.show()


def read_mu_results():
    filename = 'biased_results/excel_files/results_mu_part1.xlsx'
    df = pd.read_excel(filename, sheet_name='Right-wrong')
    df2 = pd.read_excel(filename, sheet_name='Agreements-trials')
    nonzero_data = df[df != 100000]
    mu_list = df.columns.tolist()
    
    
    percent_disagree = [1 - df2[mu].iloc[0] / df2[mu].iloc[1] for mu in mu_list]
    fig, ax = plt.subplots()
    ax.plot(mu_list, percent_disagree)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$D$')
    ax.grid(linestyle='--', axis='both')
    fig.tight_layout()
    #plt.show()
    
    filename = 'biased_results/excel_files/results_mu_part2.xlsx'
    df3 = pd.read_excel(filename, sheet_name='Right-wrong')
    filename = 'biased_results/excel_files/results_mu_part3.xlsx'
    df4 = pd.read_excel(filename, sheet_name='Right-wrong')
    
    print(df.median().idxmin(axis=1))
    num_in_first_df = len(df.columns)
    
    # df = df[df.columns[::3]]
    df = df.append(df3)
    df = df.append(df4)
    median = df.median()
    first = df.quantile(0.25)
    third = df.quantile(0.75)
    mu_list = df.columns.tolist()
    
    fig, ax = plt.subplots()
    
    lower = (median - first).array
    upper = (third - median).array
    
    lower_err = np.zeros(len(lower))
    upper_err = np.zeros(len(upper))
    for i in range(len(lower)):
        if i < num_in_first_df:
            if i % 5 == 0 or i == num_in_first_df - 1:
                lower_err[i] = lower[i]
                upper_err[i] = upper[i]
            else:
                lower_err[i] = 0
                upper_err[i] = 0
        else:
            lower_err[i] = lower[i]
            upper_err[i] = upper[i]
            
    ax.errorbar(mu_list, median, yerr=(lower_err, upper_err), elinewidth=0.8)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$t_{\rm a}^{\rm right} - t_{\rm a}^{\rm wrong}$')
    ax.grid(linestyle='--', axis='both')
    fig.tight_layout()
    
    # axin = inset_axes(ax, width='40%', height='30%', loc=4)
    # axin.errorbar(mu_list[:num_in_first_df], median.array[:num_in_first_df], yerr=(lower[:num_in_first_df], upper[:num_in_first_df]), elinewidth=0.8)
    # axin.grid(linestyle='--', axis='both')
    
    plt.show()
    
    

def test_unbalanced_triad(method=0):
    # compare between four different triads
    num_nodes = 3
    num_iter = 10000
    repetitions = 10000
    repetition10th = 10 ** (int(np.log10(repetitions)) - 1)
    
    # NOTE: does not actually support node numbers != 3 properly. For now
    tosses_list = np.ones(num_iter) * 1
    
    category_dict = {'Time': [], 'All agree': [], 'Num right': [], 'Num almost right':[], 'First right': [], 'Width': [], 'Right-wrong': []}
    
    for k in range(repetitions):
        if k % repetition10th == 0:
            print(f'Iteration {k}')
        g1 = nx.complete_graph(num_nodes)
        g1 = simulation.randomise_prior(g1)
        for edge in g1.edges():
            g1.edges[edge]['correlation'] = 1
        g1.edges[0,1]['correlation'] = -1
        
        heads_list = np.random.binomial(1, 0.5, num_iter)
        
        simulator = simulation.Simulation(g1, true_bias=0.5, tosses_per_iteration=1, bias_len=21)
        simulator.heads_list = heads_list
        simulator.tosses_list = tosses_list
        
        t = simulator.do_simulation_without_print_and_save(num_iter=num_iter, blend_method=0)
        all_agree = int(simulator.do_all_nodes_agree)
        first_right = int(simulator.is_first_node_correct)
        num_right = simulator.number_of_correct_nodes(opinion_range=(0.49,0.51), criteria='mode')
        num_almost_right = simulator.number_of_correct_nodes(opinion_range=(0.44,0.56), criteria='mode')
        width = np.abs(simulator.get_width).max()
        
        asymp = simulator.get_asymp_time
        # print(asymp)
        if first_right:
            asymp = asymp[1][0] - asymp[2][0] 
        else:
            asymp = asymp[2][0] - asymp[1][0] 
        
        category_dict['Time'].append(t)
        category_dict['All agree'].append(all_agree)
        category_dict['Num right'].append(num_right)
        category_dict['Num almost right'].append(num_almost_right)
        category_dict['First right'].append(first_right)
        category_dict['Width'].append(width)
        category_dict['Right-wrong'].append(asymp)

    
    df2 = pd.DataFrame(category_dict)
    with pd.ExcelWriter("results.xlsx") as writer:
        df2.to_excel(writer, index=False, sheet_name='By category')
 
    print(df2.describe())


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 17})
    # g = recreate_graph('output/ba3', 0)
    # edge_list = list(g.edges())
    # np.random.shuffle(edge_list)
    # midpoint = len(edge_list) // 2
    # for count, edge in enumerate(edge_list):
    #     if count < midpoint:
    #         g.edges[edge]['correlation'] = 1
    #     else:
    #         g.edges[edge]['correlation'] = -1
    # simulator3 = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=21, output_direc='output/temp2', coin_obs_file='output/ba3')
    # simulator3.do_simulation(num_iter=10000, blend_method=0)
    # analyse.DataAnalyser('output/temp2').produce_plots()
    
    two_nodes_test()
    
