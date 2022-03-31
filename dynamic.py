import numpy as np
import itertools
from simulation import put_smaller_number_first, resized_range
from numba import njit, prange

def list_of_edges(num_node):
    """Generates a list of edges for a given network
    Edge list doesn't include (i,j) where i==j, because we do not need to modify the allegiance of a node with itself"""
    return np.array(list(itertools.combinations(range(num_node),2)))

@njit
def modify_edge_weight(adj, edge_list, F, edge_weight=(-1, 1), simple_edge_weight=True, mod_probability=True, include_zero=True):
    """Takes in an adjacency matrix with a single edge modification probability of F per time step.
    'edge_list' is the list edges for the network, modified edge is to be selected randomly"""

    modification_made = False #To check if we actually modified anything

    edge_list = edge_list.copy()

    edge_list_length, _ = edge_list.shape

    if mod_probability:
        num = np.random.random()
        if num <= F:
            modification_made = True
            index_edges_to_modify = np.random.choice(np.arange(edge_list_length))
            edges_to_modify = edge_list[index_edges_to_modify]
            edge = edges_to_modify
            if simple_edge_weight and include_zero:
                roll = np.random.choice(np.arange(-1,2))
                while roll == adj[edge[0],edge[1]]:
                    roll = np.random.choice(np.arange(-1,2))
            elif simple_edge_weight:
                roll = np.random.choice(np.array([-1,1]))
                while roll == adj[edge[0],edge[1]]:
                    roll = np.random.choice(np.array([-1,1]))
            else:
                roll = np.random.uniform(-1,1)
                while roll == adj[edge[0],edge[1]]: #to prevent the roll being the same as previous edge value
                    roll = np.random.uniform(-1,1)
            adj[edge[0],edge[1]] = roll
            adj[edge[1],edge[0]] = roll

    #Not implemented for numba: Dynamics for varying fraction of edges
    # else:
    #     num_to_modify = int(F*edge_list_length)
    #     if num_to_modify!=0:
    #         modification_made = True
    #         edge_selection_counter = 0
    #         index_edges_to_modify = np.zeros(0)
    #         while edge_selection_counter < num_to_modify:
    #             index = np.random.choice(np.arange(edge_list_length))
    #             if index not in index_edges_to_modify:
    #                 index_edges_to_modify = np.append(index_edges_to_modify, index)
    #                 edge_selection_counter+=1
    #         if simple_edge_weight and include_zero:
    #             for i in index_edges_to_modify:
    #                 edge = edge_list[i]
    #                 roll = np.random.choice(np.arange(-1,2))
    #                 while roll == adj[edge[0],edge[1]]:
    #                     roll = np.random.choice(np.arange(-1,2))
    #                 adj[edge[0],edge[1]] = roll
    #                 adj[edge[1],edge[0]] = roll
    #         elif simple_edge_weight:
    #             for i in index_edges_to_modify:
    #                 edge = edge_list[i]
    #                 roll = np.random.choice(np.array([-1,1]))
    #                 while roll == adj[edge[0],edge[1]]:
    #                     roll = np.random.choice(np.array([-1,1]))
    #                 adj[edge[0],edge[1]] = roll
    #                 adj[edge[1],edge[0]] = roll
    #         else:
    #             for i in index_edges_to_modify:
    #                 edge = edge_list[i]
    #                 roll = np.random.uniform(-1,1)
    #                 while roll == adj[edge[0],edge[1]]:
    #                     roll = np.random.uniform(-1,1)
    #                 adj[edge[0],edge[1]] = roll
    #                 adj[edge[1],edge[0]] = roll

    if modification_made:
        return modification_made, adj, index_edges_to_modify
    else:
        return modification_made, adj, -1

@njit
def modify_edge_weight_F(adj, edge_list, prob, edge_weight=(-1, 1), simple_edge_weight=True, include_zero=True):
    """Takes in an adjacency matrix with a single edge modification probability of F per time step.
    'edge_list' is the list edges for the network, modified edge is to be selected randomly"""

    modification_made = False #To check if we actually modified anything

    num = np.random.random()

    if include_zero: #allowing for disconnectivity
        edge_list = edge_list.copy()

        edge_list_length, _ = edge_list.shape

        if num <= prob:
            modification_made = True
            index_edge_to_modify = np.random.choice(np.arange(edge_list_length))
            edge = edge_list[index_edge_to_modify]
            if simple_edge_weight:
                roll = np.random.choice(np.arange(-1,2))
                while roll == adj[edge[0],edge[1]]:
                    roll = np.random.choice(np.arange(-1,2))
            else:
                roll = np.random.uniform(-1,1)
                while roll == adj[edge[0],edge[1]]: #to prevent the roll being the same as previous edge value
                    roll = np.random.uniform(-1,1)
            adj[edge[0],edge[1]] = roll
            adj[edge[1],edge[0]] = roll


    else: #maintain network topology i.e. zero edges stay zero and only non-zero edges can change in value
        nonzero_edge_full = [((i, j) if i < j else (j, i)) for i, j in list(zip(*np.nonzero(adj))) if i!=j] #we do not want to modify the edge value of the node with itself hence the i!=j condition

        nonzero_edge_list = list(set(nonzero_edge_full)) #remove duplicate edges because adj is symmetric

        edge_list_length = len(nonzero_edge_list)

        if num <= prob:
            modification_made = True
            index_edge_to_modify = np.random.choice(np.arange(edge_list_length))
            edge = nonzero_edge_list[index_edge_to_modify]
            if simple_edge_weight:
                roll = -1*adj[edge[0],edge[1]]
            else:
                roll = np.random.uniform(-1,1)
                while roll == adj[edge[0],edge[1]] or roll==0: #to prevent the roll being the same as previous edge value
                    roll = np.random.uniform(-1,1)
            adj[edge[0],edge[1]] = roll
            adj[edge[1],edge[0]] = roll

    if modification_made:
        return modification_made, adj, index_edge_to_modify
    else:
        return modification_made, adj, -1


@njit #CONTINUE FROM HERE
def modify_edge_weight_P(adj, edge_list, prob, edge_weight=(-1, 1), simple_edge_weight=True, include_zero=True):
    """Takes in an adjacency matrix with a single edge modification probability of F per time step.
    'edge_list' is the list edges for the network, modified edge is to be selected randomly"""

    modification_made = False #To check if we actually modified anything

    modified_edge_count = 0

    if include_zero: #allowing for disconnectivity
        edge_list = edge_list.copy()

        for edge in edge_list:
            num = np.random.random()
            if num <= prob:
                modified_edge_count+=1
                if simple_edge_weight:
                    roll = np.random.choice(np.arange(-1,2))
                    while roll == adj[edge[0],edge[1]]:
                        roll = np.random.choice(np.arange(-1,2))
                else:
                    roll = np.random.uniform(-1,1)
                    while roll == adj[edge[0],edge[1]]: #to prevent the roll being the same as previous edge value
                        roll = np.random.uniform(-1,1)
                adj[edge[0],edge[1]] = roll
                adj[edge[1],edge[0]] = roll


    else: #maintain network topology i.e. zero edges stay zero and only non-zero edges can change in value
        nonzero_edge_full = [((i, j) if i < j else (j, i)) for i, j in list(zip(*np.nonzero(adj))) if i!=j] #we do not want to modify the edge value of the node with itself hence the i!=j condition

        nonzero_edge_list = list(set(nonzero_edge_full)) #remove duplicate edges because adj is symmetric

        edge_list_length = len(nonzero_edge_list)

        for edge in nonzero_edge_list:
            num = np.random.random()
            if num <= prob:
                modified_edge_count+=1
                if simple_edge_weight:
                    roll = -1*adj[edge[0],edge[1]]
                else:
                    roll = np.random.uniform(-1,1)
                    while roll == adj[edge[0],edge[1]] or roll==0: #to prevent the roll being the same as previous edge value
                        roll = np.random.uniform(-1,1)
                adj[edge[0],edge[1]] = roll
                adj[edge[1],edge[0]] = roll

    if modified_edge_count>0:
        modification_made = True
        
    return modification_made, adj, modified_edge_count

def triad_balance(adj):
    edge_list = list_of_edges(adj)
    negative_counter = 0
    for x, y in edge_list:
        if adj[x,y] == -1:
            negative_counter+=1
    return negative_counter
