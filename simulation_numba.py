from numba import njit, prange
import numpy as np
import scipy as sp
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from generate_graph import visualise_graph
import itertools
from read_graph import read_coin_observations, recreate_graph
from scipy.optimize import curve_fit
from scipy import stats


@njit
def do_simulation(adj, beliefs, true_bias=0.6, threshold=0.01, time_threshold=100, num_iter=10000, learning_rate=0.25):
    bias_list = np.linspace(0,1,beliefs.shape[1])
    heads_list = np.random.binomial(1, true_bias, size=num_iter)
    
    old_beliefs = beliefs.copy()
    asymp_time = np.zeros(beliefs.shape[0], dtype=np.int32)
    asymp_median = np.zeros(beliefs.shape[0], dtype=np.float64)
    num_change = 0
    
    for t in range(num_iter):
        beliefs = observe_coin(beliefs, bias_list, heads_list[t])
        beliefs = blend_pos(adj, beliefs, learning_rate=learning_rate)
        beliefs, old_beliefs, asymp_time, asymp_median, num_change = update_node_asymp(beliefs, old_beliefs, threshold, time_threshold, asymp_time, asymp_median, bias_separation=bias_list[1])
        
        if np.all(asymp_time >= time_threshold):
            return t, asymp_time, asymp_median, heads_list, num_change
            
    return 0, asymp_time, asymp_median, heads_list, num_change


@njit
def do_simulation_triad(adj, beliefs, true_bias=0.6, threshold=0.01, time_threshold=100, num_iter=10000, learning_rate=0.25):
    bias_list = np.linspace(0,1,beliefs.shape[1])
    heads_list = np.random.binomial(1, true_bias, size=num_iter)
    
    old_beliefs = beliefs.copy()
    asymp_time = np.zeros(beliefs.shape[0], dtype=np.int32)
    asymp_median = np.zeros(beliefs.shape[0], dtype=np.float64)
    num_change = 0
    
    for t in range(num_iter):
        beliefs = observe_coin(beliefs, bias_list, heads_list[t])
        beliefs = blend_pos(adj, beliefs, learning_rate=learning_rate)
        beliefs, old_beliefs, asymp_time, asymp_median, num_change = update_node_asymp(beliefs, old_beliefs, threshold, time_threshold, asymp_time, asymp_median, bias_separation=bias_list[1])
        
        if asymp_time[0] >= time_threshold and asymp_time[1] >= time_threshold:
            return t, asymp_time, asymp_median, heads_list, num_change
            
    return 0, asymp_time, asymp_median, heads_list, num_change


@njit
def do_simulation_heads_provided(adj, beliefs, heads_list, threshold=0.01, time_threshold=100):
    bias_list = np.linspace(0,1,beliefs.shape[1])
    
    old_beliefs = beliefs.copy()
    asymp_time = np.zeros(beliefs.shape[0], dtype=np.int32)
    asymp_median = np.zeros(beliefs.shape[0], dtype=np.float64)
    num_change = 0
    
    for t in range(heads_list.shape[0]):
        beliefs = observe_coin(beliefs, bias_list, heads_list[t])
        beliefs = blend_pos(adj, beliefs)
        beliefs, old_beliefs, asymp_time, asymp_median, num_change = update_node_asymp(beliefs, old_beliefs, threshold, time_threshold, asymp_time, asymp_median, bias_separation=bias_list[1])
        
        if np.all(asymp_time >= time_threshold):
            return t, asymp_time, asymp_median, num_change, beliefs
            
    return 0, asymp_time, asymp_median, num_change, beliefs
    
    
@njit
def do_simulation_with_history(adj, beliefs, true_bias=0.6, threshold=0.01, time_threshold=100, num_iter=10000, learning_rate=0.25):
    bias_list = np.linspace(0,1,beliefs.shape[1])
    heads_list = np.random.binomial(1, true_bias, size=num_iter)
    
    history = np.zeros((num_iter+1, beliefs.shape[0], beliefs.shape[1]), dtype=np.float64)
    history[0] = beliefs.copy()
    asymp_time = np.zeros(beliefs.shape[0], dtype=np.int32)
    asymp_median = np.zeros(beliefs.shape[0], dtype=np.float64)
    num_change = 0
    
    for t in range(num_iter):
        beliefs = observe_coin(beliefs, bias_list, heads_list[t])
        beliefs = blend_pos(adj, beliefs, learning_rate=learning_rate)
        beliefs, old_beliefs, asymp_time, asymp_median, num_change = update_node_asymp(beliefs, history[t], threshold, time_threshold, asymp_time, asymp_median, bias_separation=bias_list[1])
        
        history[t+1] = old_beliefs
        if np.all(asymp_time >= time_threshold):
            return t, asymp_time, asymp_median, history[:t+1], num_change
            
    
    return 0, asymp_time, asymp_median, history, num_change


@njit
def observe_coin(beliefs, bias_list, num_heads=1):
    # observe coin once and update the priors
    if num_heads == 1:
        prob = bias_list
    else:
        prob = 1 - bias_list
    
    new_beliefs = beliefs * prob
    
    norm_factor = np.sum(new_beliefs, axis=1)
    # new_beliefs = new_beliefs / norm_factor[:, np.newaxis]
    new_beliefs = new_beliefs / norm_factor.reshape((-1,1))
    
    return new_beliefs
    

@njit
def blend_pos(adj, beliefs, learning_rate=0.25):
    norm_factor = np.sum(np.abs(adj), axis=0)  # slightly faster than np.count_nonzero
    diff = beliefs - beliefs.reshape((beliefs.shape[0], 1, beliefs.shape[1]))
    diff = adj.reshape((adj.shape[0], adj.shape[1], 1)) * diff
    diff = diff.sum(axis=1)  / norm_factor.reshape((-1, 1)) * learning_rate
    new_beliefs = np.maximum(0.0, beliefs +  diff)
    return_value =  new_beliefs / new_beliefs.sum(axis=1).reshape((-1,1))
    return return_value
    
    
@njit
def update_node_asymp(beliefs, old_beliefs, threshold, time_threshold, asymp_time, asymp_median, bias_separation=0.05):
    # if a node asymps and later doesn't, reset the timer.
    num_change = 0
    diff = np.abs(beliefs - old_beliefs)
    max_old = np_max_axis1(old_beliefs)
    threshold = threshold * max_old
    threshold = threshold.reshape(-1,1)
    diff = diff <= threshold
    within_threshold = np_all_axis1(diff)
    
    old_beliefs[~within_threshold] = beliefs[~within_threshold]
    
    asymp_time[within_threshold] += 1
    asymp_time[~within_threshold] = 0
    
    new_median = np_argmax_axis1(beliefs) * bias_separation
    if np.any(asymp_median[within_threshold] != new_median[within_threshold]):
        num_change += 1
    asymp_median[within_threshold] = new_median[within_threshold]
    asymp_median[~within_threshold] = 0.0
    
    return beliefs, old_beliefs, asymp_time, asymp_median, num_change



# numba doesn't understand the axis argument for most numpy functions, so these are workarounds for them

@njit
def np_all_axis1(x):
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[0]):
        out[i] = np.all(x[i])
    return out


@njit
def np_any_axis1(x):
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[0]):
        out[i] = np.any(x[i])
    return out
    

@njit
def np_argmax_axis1(x):
    out = np.ones(x.shape[0], dtype=np.int32)
    for i in range(x.shape[0]):
        out[i] = np.argmax(x[i])
    return out

    
@njit
def np_max_axis1(x):
    out = np.ones(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = np.max(x[i])
    return out
    
    
@njit
def generate_beliefs(bias_len=21, adj_size=10):
    mean = (0.0, 1.0)
    stddev = (0.1, 0.4)
    bias_list = np.linspace(0, 1, bias_len)
    
    beliefs = np.zeros((adj_size, bias_len), dtype=np.float64)
    
    mean_draw = np.random.random((adj_size, 1)) * (mean[1] - mean[0]) + mean[0] 
    stddev_draw = np.random.random((adj_size, 1)) * (stddev[1] - stddev[0]) + stddev[0]
    
    beliefs = gaussian(bias_list, mean_draw, stddev_draw)
    beliefs_sum = beliefs.sum(axis=1)
    beliefs_sum = beliefs_sum.reshape(beliefs_sum.shape[0], 1)
    beliefs = beliefs / beliefs_sum
    return beliefs

    
@njit
def gaussian(bias_list, mean=0.5, stddev=0.5):
    #return np.exp(-4 * np.log(2) * ((bias_list - mean) / stddev) ** 2)
    return np.exp(-((bias_list-mean) / stddev)**2 / 2)


@njit
def ba_graph(n=10,m=5):
    # based on networkx's barabasi_albert_graph. Not fully tested!
    def get_degree(adj):
        return np.sum(np.abs(adj), axis=0)
        
    adj = np.zeros((n,n), dtype=np.int32)
    adj[m:m+1, :m+1] = 1
    adj[:m+1, m:m+1] = 1
    adj[m,m] = 0
    
    for i in range(m+1, n):
        current_degree = get_degree(adj[:i, :i])
        chosen_nodes = []
        for j in range(m):
            prob = []
            for node, degree in enumerate(current_degree):
                if node not in chosen_nodes:
                    for _ in range(degree):
                        prob.append(node)
            choice = np.random.choice(np.array(prob))
            chosen_nodes.append(choice)
            adj[i,choice] = 1
            adj[choice, i] = 1
    
    return adj
    


def get_adj_from_nx(g):
    n = g.number_of_nodes()
    adj = np.zeros((n,n), dtype=np.int32)
    for u, v, correlation in g.edges(data='correlation'):
        adj[u, v] = correlation
        adj[v, u] = correlation
        
    # for u, v in g.edges():
    #     adj[u, v] = 1
    #     adj[v, u] = 1
        
    return adj
    
    
def get_nx_from_adj(adj):
    g = nx.Graph()
    g.add_nodes_from(range(adj.shape[0]))
    
    indices1, indices2 = np.triu_indices(adj.shape[0], 1)
    indices = zip(indices1, indices2)
    
    for index in indices:
        if adj[index] != 0:
            g.add_edge(index[0], index[1], correlation=adj[index])
    return g


def test_graph_balance(adj):
    """	
    Return values:	
    0: Unbalanced	
    1: Weakly balanced	
    2: Strongly balanced
    
    adj must be a numpy array
    """
    adj2 = adj.copy()
    np.fill_diagonal(adj2, 1)  # diagonal entries must be >0 for the below trick to work
    group_id = np.full(adj.shape[0], adj.shape[0], dtype=np.int32)
    
    # group friends into the same group
    checked_nodes = set()
    for i in range(adj.shape[0]):
        if i in checked_nodes:
            continue
        group_id[i] = i
        queued_friends = {i}  # this is a set
        while queued_friends:
            current_node = queued_friends.pop()
            
            friends = np.nonzero(adj2[current_node] > 0)[0]
            new_friends = [j for j in friends if not j in checked_nodes]
            for friend in new_friends:
                group_id[friend] = i
            queued_friends.update(new_friends)
            checked_nodes.add(current_node)
        
    # now check for enemies. if two nodes in the same group are enemies, then the network is unbalanced
    # for i in range(adj.shape[0]-1):
    #     enemies = adj2[i] < 0
    #     has_contradiction = np.any(group_id[i] == group_id[enemies])
    #     if has_contradiction:
    #         return 0
    
    enemies = adj2 < 0
    has_contradiction = np.any((group_id.reshape(-1,1) == group_id)[enemies])
    if has_contradiction:
        return 0
        
    # now check if we can minimise the number of unique group id
    no_change = False
    while not no_change:
        no_change = True
        unique_id = np.unique(group_id)
        combo = []
        for i in range(unique_id.shape[0]):
            for j in range(i+1, unique_id.shape[0]):
                combo.append((unique_id[i], unique_id[j]))
        # combo = itertools.combinations(unique_id, 2)
        for u, v in combo:
            nodes1 = np.nonzero(group_id == u)[0]
            nodes2 = np.nonzero(group_id == v)[0]
            can_merge = True
            for i in nodes1:
                for j in nodes2:
                    if adj2[i,j] < 0:
                        can_merge = False
                        break
                if not can_merge:
                    break
            
            if can_merge:
                no_change = False
                group_id[nodes2] = u
                break
                
    # the graph is not unbalanced. now check the number of unique group ids to determine the number of clusters
    num_clusters = np.unique(group_id).shape[0]
    if num_clusters <= 2:
        return 2
    return 1
    # return np.unique(group_id).shape[0]


def test_first_wrong():
    bias_len = 21
    bias_list = np.linspace(0,1,bias_len)
    # results_dict = {'Diff': [], 'Median diff': []}
    results_dict = {'Right': [], 'Wrong': [], 'Median diff': []}
    true_bias = 0.6
    
    max_time = 10000
    repetitions = 100000
    repetition10th = repetitions // 100
    time_threshold = 100
    
    num_both_wrong = 0
    num_both_agree = 0
    num_no_asymp = 0
    num_both_same = 0
    num_both_correct = 0
    num_both_wrong_apparent = 0
    num_first_correct_apparent = 0
    
    adj = np.array([[0,-1],[-1,0]])
    total_num_change = 0
    
    for i in range(repetitions):
        if i % repetition10th == 0:
            print(i)
            
        beliefs = generate_beliefs(adj_size=2)
        
        time, asymp_time, asymp_median, heads_list, num_change = do_simulation(adj, beliefs, num_iter=max_time, threshold=0.01, true_bias=true_bias, time_threshold=time_threshold)
        if asymp_median[0] == asymp_median[1]:
            num_both_agree += 1
            continue
        
        total_num_change += num_change
        
        node1_correct = true_bias - 0.01 < asymp_median[0] < true_bias + 0.01
        node2_correct = true_bias - 0.01 < asymp_median[1] < true_bias + 0.01
        
        if node1_correct and node2_correct:
            num_both_correct += 1
        elif node1_correct:
            correct_node = 0
        elif node2_correct:
            correct_node = 1
        else:
            num_both_wrong += 1
            apparent_bias = np.sum(heads_list[:time]) / time
            if apparent_bias != true_bias:
                num_both_wrong_apparent += 1
            continue
        
        if time == 0 or asymp_time[0] < time_threshold or asymp_time[1] < time_threshold:
            num_no_asymp += 1
            
        if asymp_time[0] == asymp_time[1]:
            num_both_same += 1
        
        time_diff = asymp_time[1-correct_node] - asymp_time[correct_node]
        if time_diff < 0:
            wrong_asymp_time = time - asymp_time[1-correct_node]
            apparent_bias = np.sum(heads_list[:wrong_asymp_time]) / time
            if apparent_bias - 0.05 < asymp_median[1-correct_node] < apparent_bias + 0.05:
                num_first_correct_apparent += 1
            
        results_dict['Right'].append(time - asymp_time[correct_node])
        results_dict['Wrong'].append(time - asymp_time[1-correct_node])
        results_dict['Median diff'].append(asymp_median[1-correct_node] - asymp_median[correct_node])
        
    print(f'Both agree          : {num_both_agree}')
    print(f'Both wrong          : {num_both_wrong}')
    print(f'Num no asymp        : {num_no_asymp}')
    print(f'Num change          : {total_num_change}')
    print(f'Num same            : {num_both_same}')
    print(f'Num both correct    : {num_both_correct}')
    print(f'Num wrong apparent  : {num_both_wrong_apparent}')
    print(f'Num correct apparent: {num_first_correct_apparent}')
    
    df = pd.DataFrame(results_dict)
    df2 = pd.DataFrame({'Num trials': [repetitions]})
    with pd.ExcelWriter("results_pair.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        df2.to_excel(writer, index=False, sheet_name='Number of trials')

    print(df.describe())
    print(df[df['Diff'] > 0].count())
    print(df[df['Diff'] == 0].count())
    print(df[df['Diff'] < 0].count())
    print(df.quantile((0.05,0.95)))
    
    plt.rcParams.update({'font.size': 17})
    fig, ax = plt.subplots()
    ax.hist(df['Diff'], bins='auto')
    ax.set_xlabel(r'$t_a^{\rm right} - t_a^{\rm wrong}$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(df['Median diff'], bins=np.arange(np.min(df['Median diff']) - 0.025, np.max(df['Median diff']) + 0.025, 0.05))
    ax.set_xlabel(r'$\langle \theta \rangle ^{\rm right} - \theta_0$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    
    plt.show()


def test_first_wrong_mu():
    bias_len = 21
    true_bias = 0.6
    bias_list = np.linspace(0,1,bias_len)
    # mu_list = np.arange(0.0001, 0.0051, 0.0001)
    mu_list = np.arange(0.006, 0.051, 0.001)
    # mu_list = np.arange(0.06, 0.51, 0.01)
    time_diff_dict = {}  # measure as right - wrong
    theta_diff_dict = {}  # measure as right - wrong
    num_both_wrong_dict = {}
    for mu in mu_list:
        time_diff_dict[mu] = []
        theta_diff_dict[mu] = []
        num_both_wrong_dict[mu] = [0]
    num_agree_dict = {}
    for mu in mu_list:
        num_agree_dict[mu] = [0,0]  # [num_agree, total_sims]
    
    max_time = 10000
    repetitions = 1000
    adj = np.array([[0,-1],[-1,0]])
    
    for mu in mu_list:
        print(mu)
        i = 0
        while i < repetitions:
            beliefs = generate_beliefs(adj_size=2)
            
            time, asymp_time, asymp_median, _, _ = do_simulation(adj, beliefs, num_iter=max_time, threshold=0.01, true_bias=true_bias, learning_rate=mu, time_threshold=100)
            
            if asymp_median[0] == asymp_median[1]:
                num_agree_dict[mu][0] += 1
                num_agree_dict[mu][1] += 1
                continue
                
            if true_bias - 0.01 < asymp_median[0] < true_bias + 0.01:
                correct_node = 0
            elif true_bias - 0.01 < asymp_median[1] < true_bias + 0.01:
                correct_node = 1
            else:
                num_both_wrong_dict[mu][0] += 1
                continue
            time_diff = asymp_time[1-correct_node] - asymp_time[correct_node]  # remember that small asymp_time corresponds to later $t_a$
            time_diff_dict[mu].append(time_diff)
            theta_diff_dict[mu].append(asymp_median[correct_node] - asymp_median[1-correct_node])
            num_agree_dict[mu][1] += 1
            i += 1
        
    df = pd.DataFrame(time_diff_dict)
    df2 = pd.DataFrame(num_agree_dict)
    df3 = pd.DataFrame(theta_diff_dict)
    df4 = pd.DataFrame(num_both_wrong_dict)
    with pd.ExcelWriter("results_mu_part2.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Right-wrong')
        df2.to_excel(writer, index=False, sheet_name='Agreements-trials')
        df3.to_excel(writer, index=False, sheet_name='Right-wrong-median')
        df4.to_excel(writer, index=False, sheet_name='Both wrong')
    
    percent_disagree = [1 - df2[mu].iloc[0] / df2[mu].iloc[1] for mu in mu_list]
    fig, ax = plt.subplots()
    ax.plot(mu_list, percent_disagree)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'Percentage of disagreements')
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


def test_unbalanced_triad():
    bias_len = 21
    bias_list = np.linspace(0,1,bias_len)
    results_dict = {'Diff': []}
    theta_diff_dict = {'Diff': []}
    true_bias = 0.6
    
    max_time = 15000
    repetitions = 100000
    repetition10th = repetitions // 100
    time_threshold = 100
    
    num_both_wrong = 0
    num_both_agree = 0
    num_no_asymp = 0
    num_both_same = 0
    
    adj = np.array([[0,-1, 1],[-1,0, 1], [1,1,0]])
    total_num_change = 0
    
    for i in range(repetitions):
        if i % repetition10th == 0:
            print(i)
            
        beliefs = generate_beliefs(adj_size=3)
        
        time, asymp_time, asymp_median, _, num_change = do_simulation_triad(adj, beliefs, num_iter=max_time, threshold=0.01, true_bias=true_bias, time_threshold=time_threshold)
        if asymp_median[0] == asymp_median[1]:
            num_both_agree += 1
            continue
        
        total_num_change += num_change
        
        if true_bias - 0.01 < asymp_median[0] < true_bias + 0.01:
            correct_node = 0
        elif true_bias - 0.01 < asymp_median[1] < true_bias + 0.01:
            correct_node = 1
        else:
            num_both_wrong += 1
            continue
        
        if (asymp_time[0] < time_threshold or asymp_time[1] < time_threshold) and num_change > 0:
            num_no_asymp += 1
            print('Nyehehe')
            
        if asymp_time[0] == asymp_time[1]:
            num_both_same += 1
        
        if correct_node == 0:
            wrong_node = 1
        else:
            wrong_node = 0
        
        time_diff = asymp_time[wrong_node] - asymp_time[correct_node]
        results_dict['Diff'].append(time_diff)
        
    print(f'Both agree  : {num_both_agree}')
    print(f'Both wrong  : {num_both_wrong}')
    print(f'Num no asymp: {num_no_asymp}')
    print(f'Num change  : {total_num_change}')
    print(f'Num same    : {num_both_same}')
    df = pd.DataFrame(results_dict)
    df2 = pd.DataFrame({'Num trials': [repetitions]})
    with pd.ExcelWriter("results_triad.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        df2.to_excel(writer, index=False, sheet_name='Number of trials')

    print(df.describe())
    print(df[df['Diff'] > 0].count())
    print(df[df['Diff'] == 0].count())
    print(df[df['Diff'] < 0].count())
    print(df.quantile((0.05,0.95)))
    
    plt.rcParams.update({'font.size': 17})
    fig, ax = plt.subplots()
    ax.hist(df['Diff'], bins='auto')
    ax.set_xlabel(r'$t_a^{\rm right} - t_a^{\rm wrong}$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    plt.show()

 
def read(file='output/results_pair.xlsx'):
    df = pd.read_excel(file)
    print(df.describe())
    print(df.mode())
    
    new_df = df['Right'] - df['Wrong']
    print(new_df.describe())
    print(new_df[new_df > 0].count())
    print(new_df[new_df < 0].count())
    print(new_df[new_df == 0].count())
    
    fig_size = (4.5,4)
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.hist(new_df[new_df < 5000], bins='auto')
    ax.set_xlabel(r'$t_a^{\rm right} - t_a^{\rm wrong}$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    # ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.hist(df[df['Right'] < 5000]['Right'], bins='auto')
    ax.set_xlabel(r'$t_a^{\rm right}$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    # ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.hist(df[df['Wrong'] < 1000]['Wrong'], bins='auto')
    ax.set_xlabel(r'$t_a^{\rm wrong}$')
    ax.set_ylabel('Count')
    ax.grid(linestyle='--', axis='both')
    # ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(df['Median diff'], bins=np.arange(min(df['Median diff']) - 0.025, max(df['Median diff']) + 0.025, 0.05))
    ax.set_xlabel(r'$\langle \theta \rangle ^{\rm wrong} - \langle \theta \rangle ^{\rm wrong}$')
    ax.set_ylabel('Counts')
    ax.grid(linestyle='--', axis='both')
    # ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    plt.show()

def test_ba():
    n = 100
    m = 10

    num_graphs = 1
    key_unbalanced = 0
    key_weak = 1
    key_strong = 2
    num_computed_list = np.zeros(3, dtype=int)
    results = [[], [], []]
    
    count = 0
    num_failed = 0
    
    while np.any(num_computed_list < num_graphs):
        adj = get_ba_adj_random_sign(n,m)
        balance = test_graph_balance(adj)
        if num_computed_list[balance] >= num_graphs:
            num_failed += 1
            if num_failed % 1000 == 0:
                print(f'Failed: {num_failed}')
            continue
        
        num_failed = 0
        beliefs = generate_beliefs(adj_size=n)
        t, *_ = do_simulation(adj=adj, beliefs=beliefs, num_iter=10000)
        results[balance].append(t)
        num_computed_list[balance] += 1
        count += 1
        print(count)
        if num_computed_list[balance] == num_graphs:
            print(f'{balance} done.')
        
    df = pd.DataFrame(results, columns=['Unbalanced', 'Weak', 'Strong'])
    print('')
    print(df[df > 0].describe())


def get_ba_adj_random_sign(n=100,m=3):
    # note: include seed=100 for reproducability 
    g = nx.barabasi_albert_graph(n,m)
    for edge in g.edges():
        g.edges[edge]['correlation'] = 1 if np.random.random() < 0.5 else -1
    adj = get_adj_from_nx(g)
    return adj


def get_clustered_complete_graph(n=10, k=2):
    if k <= 0:
        raise ValueError('Number of clusters must be more than 0')
    if k > n:
        raise ValueError('Number of clusters cannot exceed number of nodes')
        
    adj = np.ones((n,n), dtype=int)
    m = n-k
    cluster_sizes = np.ones(k, dtype=int) + np.random.multinomial(m, np.ones(k)/k) 
    
    nodes_id = []
    for count, i in enumerate(cluster_sizes):
        nodes_id.extend([count] * i)
    nodes_id = np.array(nodes_id)
    np.random.shuffle(nodes_id)
    
    a = nodes_id.reshape(-1,1) == nodes_id
    
    adj[~a] = -1
    return adj
    
def get_clustered_ba_graph(n=10, m=3, k=2):
    """
    n: number of nodes
    m: attachment parameter
    k: number of clusters
    """
    
    # adj_ba = ba_graph(n,m)
    # adj_cluster = get_clustered_complete_graph(n,k)
    # return adj_ba * adj_cluster
    
    g = nx.barabasi_albert_graph(n,m)
    for edge in g.edges():
        g.edges[edge]['correlation'] = 1
    adj = get_adj_from_nx(g)
    
    cluster_sizes = np.ones(k, dtype=int) + np.random.multinomial(n-k, np.ones(k)/k)
    nodes_id = []
    for count, i in enumerate(cluster_sizes):
        nodes_id.extend([count] * i)
    nodes_id = np.array(nodes_id)
    np.random.shuffle(nodes_id)
    
    is_same_group = nodes_id.reshape(-1,1) == nodes_id
    is_connected = adj == 1
    adj[(~is_same_group) & is_connected] = -1
    return adj
    
    
def test_ba_hist_opponents():
    heads_list = np.array(read_coin_observations('output/ba3/data/observations.txt')[0])
    g = recreate_graph('output/ba2')
    adj = get_adj_from_nx(g)
    repetitions = 1000
    
    @njit
    def parallel_sims(adj, n=1000):
        median_list = []
        enemies_agree_list = []
        percent_correct_list = []
        correct_times = []
        wrong_times = []
        triu = np.triu(adj, 1)
        enemies = triu < 0
        indices = np.nonzero(enemies)
        indices = list(zip(indices[0], indices[1]))
        
        degree_list = np.sum(np.abs(adj), axis=1)
        for i in range(n):
            print(i)
            beliefs = generate_beliefs(adj_size=100)
            t, asymp_time, asymp_median, _, _ = do_simulation(adj, beliefs)
            median_list.extend(asymp_median)
            
            current_enemies_agree = []
            for u, v in indices:
                if asymp_median[u] == asymp_median[v]:
                    #corrcoef = np.corrcoef(beliefs[u], beliefs[v])[0,1]
                    prior1 = beliefs[u]
                    prior2 = beliefs[v]
                    a = np.max(np.abs(prior1-prior2))
                    b = np.max(np.maximum(prior1,prior2))
                    corrcoef = a/b
                    if asymp_time[u] > asymp_time[v]:  # put whoever asymp first, first
                        current_enemies_agree.append((asymp_median[u], degree_list[u], degree_list[v], corrcoef))
                    else:
                        current_enemies_agree.append((asymp_median[u], degree_list[v], degree_list[u], corrcoef))
            if current_enemies_agree:
                enemies_agree_list.append(current_enemies_agree)
                
            correct_nodes = (asymp_median < 0.61) & (asymp_median > 0.59)
            percent_correct_list.append(np.count_nonzero(correct_nodes))
            for node, correct in enumerate(correct_nodes):
                if correct:
                    correct_times.append(t - asymp_time[node])
                else:
                    wrong_times.append(t - asymp_time[node])
            
        return median_list, enemies_agree_list, percent_correct_list, correct_times, wrong_times
    
    results_list, enemies_agree_list, percent_correct_list, correct_times, wrong_times = parallel_sims(adj, repetitions)
        
    np.savetxt('ba_median.txt', results_list)
    np.savetxt('ba_percent_correct.txt', percent_correct_list, fmt='%i')
    np.savetxt('ba_time_correct.txt', correct_times, fmt='%i')
    np.savetxt('ba_time_wrong.txt', wrong_times, fmt='%i')
    with open('enemy_agree.txt', 'w+') as f:
        for line in enemies_agree_list:
            line_to_write = ''
            for i in line:
                line_to_write += f'{i[0]:.2f} {i[1]} {i[2]} {i[3]}, '
            line_to_write = line_to_write[:-2]
            line_to_write += '\n'
            f.write(line_to_write)
             
    
    fig, ax = plt.subplots()
    ax.hist(results_list, bins=np.arange(np.min(results_list) - 0.025, np.max(results_list) + 0.025, 0.05), weights=np.ones_like(results_list) / len(results_list))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$\langle \theta \rangle$')
    ax.set_ylabel(r'Counts (%)')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(percent_correct_list, bins=np.arange(np.min(percent_correct_list) - 0.5, np.max(percent_correct_list) + 0.5, 1))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'Number of correct agents')
    ax.set_ylabel(r'Counts')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(correct_times, bins='auto')
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'Correct times')
    ax.set_ylabel(r'Counts')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(wrong_times, bins='auto')
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'Wrong times')
    ax.set_ylabel(r'Counts')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    print(pd.DataFrame({'Correct times': correct_times}).describe())
    print(pd.DataFrame({'Wrong times': wrong_times}).describe())
    
    
    plt.show()


def read_ba_hist():
    num_agree_list = []
    degree_diff_list = []
    degree_list = []
    corrcoef_list = []
    percent_correct_list = []
    
    # with open('enemy_agree.txt', 'r') as f:
    #     lines = f.readlines()
    
    # for line in lines:
    #     data = line.split(', ')
    #     num_agree = 0
    #     for individual_data in data:
    #         median, degree1, degree2, corrcoef = individual_data.split()
    #         median = float(median)
    #         degree1 = int(degree1)
    #         degree2 = int(degree2)
    #         corrcoef = float(corrcoef)
    #         degree_diff_list.append(abs(degree1 - degree2))
    #         degree_list.append(degree2)
    #         degree_list.append(degree1)
    #         corrcoef_list.append(corrcoef)
    #         num_agree += 1
    #     num_agree_list.append(num_agree)
        
    # print(pd.DataFrame({'Num agree': num_agree_list}).describe())
    # print(pd.DataFrame({'degree_diff_list': degree_diff_list}).describe())
    # print(pd.DataFrame({'degree_list': degree_list}).describe())
    # print(pd.DataFrame({'corrcoef': corrcoef_list}).describe())
    # 
    # g = recreate_graph('output/ba2')
    # degree = [g.degree(n) for n in g.nodes()]
    # print(pd.DataFrame({'degree': degree}).describe())
    
    # fig, ax = plt.subplots()
    # ax.hist(corrcoef_list, bins='auto')
    # ax.grid(linestyle='--', axis='both')
    # fig.tight_layout()
    # plt.show()
    
    # fig, ax = plt.subplots()
    # ax.hist(degree_diff_list, bins='auto')
    # ax.grid(linestyle='--', axis='both')
    # fig.tight_layout()
    # plt.show()
    
    medians = np.loadtxt('ba_median.txt')
    percent_correct_list = np.loadtxt('ba_percent_correct.txt', dtype=int)
    correct_times = np.loadtxt('ba_time_correct.txt', dtype=int)
    wrong_times = np.loadtxt('ba_time_wrong.txt', dtype=int)
    
    print(pd.DataFrame({'Correct times': correct_times}).describe())
    print(pd.DataFrame({'Wrong times': wrong_times}).describe())
    print(pd.DataFrame({'Medians': medians}).describe())
    print(pd.DataFrame({'Num correct': percent_correct_list}).describe())
    
    print(stats.mode(wrong_times))
    
    
    wrong_times = wrong_times[wrong_times < 1000]
    
    fig, ax = plt.subplots()
    ax.hist(medians, bins=np.arange(np.min(medians) - 0.025, np.max(medians) + 0.025, 0.05))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$\langle \theta \rangle$')
    ax.set_ylabel(r'Counts')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(percent_correct_list, bins=np.arange(np.min(percent_correct_list) - 0.5, np.max(percent_correct_list) + 0.5, 1))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'Number of $\langle \theta \rangle \rightarrow \theta_0$ per simulation')
    ax.set_ylabel(r'Counts')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(correct_times, bins='auto')
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$t_{\rm a}^{\rm right}$')
    ax.set_ylabel(r'Counts')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(wrong_times, bins='auto')
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$t_{\rm a}^{\rm wrong}$')
    ax.set_ylabel(r'Counts')
    #ax.ticklabel_format(scilimits=(-3,3))
    fig.tight_layout()
    
    plt.show()


def test_ba_num_wrong_vs_n():
    repetitions = 100
    n_range = np.arange(10, 101, 5)
    num_correct_dict = {}
    correct_times_dict = {}
    wrong_times_dict = {}
    
    for n in n_range:
        adj = get_ba_adj_random_sign(n)
        triu = np.triu(adj, 1)
        enemies = triu < 0
        indices = np.nonzero(enemies)
        indices = list(zip(indices[0], indices[1]))
        
        num_correct_list = []
        correct_times = []
        wrong_times = []
        
        for i in range(repetitions):
            beliefs = generate_beliefs(adj_size=n)
            t, asymp_time, asymp_median, _, _ = do_simulation(adj, beliefs, heads_list)
            
            correct_nodes = (asymp_median < 0.61) & (asymp_median > 0.59)
            num_correct_list.append(np.count_nonzero(correct_nodes))
            for node, correct in enumerate(correct_nodes):
                if correct:
                    correct_times.append(t - asymp_time[node])
                else:
                    wrong_times.append(t - asymp_time[node])
                    
        num_correct_dict[n] = num_correct_list
        correct_times_dict[n] = correct_times
        wrong_times_dict[n] = wrong_times
        
    df = pd.DataFrame(num_correct_dict)
    df2 = pd.DataFrame(correct_times_dict)
    df3 = pd.DataFrame(wrong_times_dict)
    with pd.ExcelWriter("ba_vs_n.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Num correct')
        df2.to_excel(writer, index=False, sheet_name='Right times')
        df3.to_excel(writer, index=False, sheet_name='Wrong times')


def test_ba_balance():
    @njit
    def test_asymp(adj, asymp_time):
        did_not_asymp = asymp_time < 100
        for count, asymp in enumerate(did_not_asymp):
            if asymp:
                if np.count_nonzero(adj[count] > 0) == 0:
                    print('Ye')
                    break

    repetitions = 1000
    num_no_asymp = {'Strong': np.ones(repetitions, dtype=int), 'Weak': np.ones(repetitions, dtype=int), 'Unbalanced': np.ones(repetitions, dtype=int)}
    asymp_times = {'Strong': np.ones(repetitions, dtype=int), 'Weak': np.ones(repetitions, dtype=int), 'Unbalanced': np.ones(repetitions, dtype=int)}
    
    n = 100
    m = 3
    for i in range(repetitions):
        print(i)
        adj = get_ba_adj_random_sign(n,m)
        if test_graph_balance(adj) != 0:
            print('Ay')
            continue
        beliefs = generate_beliefs(adj_size=100)
        
        t, asymp_time, asymp_median, heads_list, num_change = do_simulation(adj, beliefs)
        num_no_asymp['Unbalanced'][i] = np.count_nonzero(asymp_time < 100)
        asymp_times['Unbalanced'][i] = t
        
        test_asymp(adj, asymp_time)
            
    
    print('Unbalanced done')
        
    for i in range(repetitions):
        print(i)
        adj = get_clustered_ba_graph(n,m,k=2)
        if test_graph_balance(adj) != 2:
            print('Ayy')
            continue
        beliefs = generate_beliefs(adj_size=100)
        
        t, asymp_time, asymp_median, heads_list, num_change = do_simulation(adj, beliefs)
        num_no_asymp['Strong'][i] = np.count_nonzero(asymp_time < 100)
        asymp_times['Strong'][i] = t
        
        test_asymp(adj, asymp_time)
        
    print('Strong done')
        
    for i in range(repetitions):
        print(i)
        k = np.random.randint(3,101)
        adj = get_clustered_ba_graph(n,m,k=k)
        if test_graph_balance(adj) != 1:
            print('Ayyy')
            continue
        beliefs = generate_beliefs(adj_size=100)
        
        t, asymp_time, asymp_median, heads_list, num_change = do_simulation(adj, beliefs)
        num_no_asymp['Weak'][i] = np.count_nonzero(asymp_time < 100)
        asymp_times['Weak'][i] = t
        
        test_asymp(adj, asymp_time)
    
    df = pd.DataFrame(num_no_asymp)
    df2 = pd.DataFrame(asymp_times)
    with pd.ExcelWriter("balance_big.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Num no asymp')
        df2.to_excel(writer, index=False, sheet_name='Asymp times')
        
    print(df.describe())
    print(df2.describe())
    
    
def test_ba_mixed():
    repetitions = 1000
    num_no_asymp = np.ones(repetitions, dtype=int)
    asymp_times = np.ones(repetitions, dtype=int)
    g = recreate_graph('output/ba3')
    adj = get_adj_from_nx(g)
    
    for i in range(repetitions):
        print(i)
        beliefs = generate_beliefs(adj_size=100)
        
        t, asymp_time, asymp_median, heads_list, num_change = do_simulation(adj, beliefs)
        num_no_asymp[i] = np.count_nonzero(asymp_time < 100)
        asymp_times[i] = t
    
    df = pd.DataFrame({'Num no asymp': num_no_asymp, 'Asymp time': asymp_times})
    with pd.ExcelWriter("mixed_only.xlsx") as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
        
    print(df.describe())


def test_sus():
    beliefs = None
    heads_list = None
    adj = None
    
    while True:
        adj = get_clustered_ba_graph(100,3,2)
        beliefs = generate_beliefs(adj_size=100)
        
        t, asymp_time, asymp_median, heads_list, num_change = do_simulation(adj, beliefs)
        if np.any(asymp_time < 100):
            break
    
    g = get_nx_from_adj(adj)
    
    for n in g.nodes():
        g.nodes[n]['prior'] = beliefs[n]
    
    import simulation, analyse
    folder = 'output/test_sus2'
    simulator = simulation.Simulation(g, true_bias=0.6, tosses_per_iteration=1, bias_len=21, output_direc=folder)
    simulator.heads_list = heads_list
    simulator.tosses_list = np.ones(10000, dtype=int)
    simulator.do_simulation(num_iter=10000, blend_method=0, learning_rate=0.25)
    analyser = analyse.DataAnalyser(folder)
    analyser.produce_plots()


def read_ba_mixed():
    df = pd.read_excel('output/mixed_only.xlsx')
    print(df.describe())
    
    fig, ax = plt.subplots()
    ax.hist(df['Num no asymp'], bins=np.arange(min(df['Num no asymp'])-0.5, max(df['Num no asymp']) + 1.5, 1))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Counts')
    fig.tight_layout()
    plt.show()


def read_ba_balance():
    df = pd.read_excel('output/balance_big.xlsx', sheet_name='Num no asymp')
    df2 = pd.read_excel('output/balance_big.xlsx', sheet_name='Asymp times')
    
    nonzero_df = df[df > 0]
    nonzero_df2 = df2[df2 > 0]
    
    print(nonzero_df.describe())
    print(nonzero_df2.describe())
    print(nonzero_df.quantile(0.9))
    
    
    fig, ax = plt.subplots()
    ax.hist(nonzero_df['Strong'], bins=np.arange(np.nanmin(nonzero_df['Strong'])-0.5, np.nanmax(nonzero_df['Strong']) + 1.5, 1))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Counts')
    ax.set_title('Strongly balanced')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(nonzero_df['Weak'], bins=np.arange(np.nanmin(nonzero_df['Weak'])-0.5, np.nanmax(nonzero_df['Weak']) + 1.5, 1))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Counts')
    ax.set_title('Weakly balanced')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(nonzero_df['Unbalanced'], bins=np.arange(np.nanmin(nonzero_df['Unbalanced'])-0.5, np.nanmax(nonzero_df['Unbalanced']) + 1.5, 1))
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Counts')
    ax.set_title('Unbalanced')
    fig.tight_layout()
    
    plt.show()
    
    fig, ax = plt.subplots()
    ax.hist(nonzero_df2['Strong'], bins='auto')
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$t_{\rm a}$')
    ax.set_ylabel('Counts')
    ax.set_title('Strongly balanced')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(nonzero_df2['Weak'], bins='auto')
    ax.grid(linestyle='--', axis='both')
    ax.set_xlabel(r'$t_{\rm a}$')
    ax.set_ylabel('Counts')
    ax.set_title('Weakly balanced')
    fig.tight_layout()
    
    plt.show()


def example_balance():
    def make_graph_clustered(g,k):
        for edge in g.edges():
            g.edges[edge]['correlation'] = 1
        adj = get_adj_from_nx(g)
        n = adj.shape[0]
        
        cluster_sizes = np.ones(k, dtype=int) + np.random.multinomial(n-k, np.ones(k)/k)
        nodes_id = []
        for count, i in enumerate(cluster_sizes):
            nodes_id.extend([count] * i)
        nodes_id = np.array(nodes_id)
        np.random.shuffle(nodes_id)
        
        is_same_group = nodes_id.reshape(-1,1) == nodes_id
        is_connected = adj == 1
        adj[(~is_same_group) & is_connected] = -1
        
        return adj
    
        
    g = nx.barabasi_albert_graph(10,2, seed=150)
    
    adj = make_graph_clustered(g,2)
    visualise_graph(get_nx_from_adj(adj), True)
    np.savetxt('test1.txt', adj, fmt='%i')
    
    adj = make_graph_clustered(g,3)
    visualise_graph(get_nx_from_adj(adj), True)
    np.savetxt('test2.txt', adj, fmt='%i')
    
    for edge in g.edges():
        g.edges[edge]['correlation'] = 1 if np.random.random() < 0.5 else -1
        
    visualise_graph(g, True)
    np.savetxt('test3.txt', adj, fmt='%i')
    

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 17})
    #test_ba_balance()
    # g = recreate_graph('output/ba3')
    # 
    # num_negative = 0
    # num_positive = 0
    # for _, _, edge_weight in g.edges(data='correlation'):
    #     if edge_weight < 0:
    #         num_negative += 1
    #     else:
    #         num_positive += 1
    # print(num_negative)
    # print(num_positive)
    
    read_ba_balance()