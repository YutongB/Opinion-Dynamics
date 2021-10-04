from pyvis.network import Network
from read_graph import recreate_graph
import numpy as np


"""
Plot pretty graphs using pyvis. Generated Figure 9 using this.
"""


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
        return None
        
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
                
    return group_id

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

def turn_nx_into_nt(g):
    nt = Network(height='700px', width='700px')
    node_ids = test_graph_balance(get_adj_from_nx(g))
    for n in g.nodes():
        nt.add_node(n, size=5*g.degree[n]+5, label=str(g.degree[n]), color='red' if node_ids[n] == 0 else 'blue')
    for u, v, correlation in g.edges(data='correlation'):
        if correlation > 0:
            nt.add_edge(u, v, color='blue')
        else:
            nt.add_edge(u, v, color='red')
    return nt


def main():
    folder = 'output/test_sus'
    g = recreate_graph(folder)
    
    nt = turn_nx_into_nt(g)
    #nt.barnes_hut(spring_strength=0.006)
    nt.barnes_hut()
    nt.set_edge_smooth('discrete')
    nt.show_buttons('physics')
    nt.show('nx.html')
    

if __name__ == '__main__':
    main()