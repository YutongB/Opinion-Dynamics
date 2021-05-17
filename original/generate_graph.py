import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from read_graph import recreate_graph
import itertools


# convinience function for testing cycles
def group_list_elements_into_tuples(l, size=2, cycle_back=True):
    # length = len(l)
    # new_list = []
    # for i in range(length):
    #     if i + size - 1 >= length:
    #         if cycle_back:
    #             temp_list = [j for j in l[i:i+size-1]]
    #             temp_list.append(l[0])
    #             new_list.append(temp_list)
    #         break
    #     new_list.append([j for j in l[i:i+size]])
    # return new_list
    
    length = len(l)
    for i in range(length):
        if i + size - 1 >= length:
            if cycle_back:
                temp_list = [j for j in l[i:i+size-1]]
                temp_list.append(l[0])
                yield temp_list
            break
        yield [j for j in l[i:i+size]]


def core_periphery(n=10, m=10, p=0.5):
    """
    n: Number of cores
    m: Number of peripheries
    p: Probability of a core-periphery connection
    """
    adj_list = []

    for i in range(n + m):
        count = i
        adj_entry = '{}'.format(count)

        count += 1

        if i < n:

            while count < n:
                adj_entry += ' {}'.format(count)
                count += 1

            while count < n + m:
                if np.random.random() < p:
                    adj_entry += ' {}'.format(count)
                count += 1

        adj_list.append(adj_entry)

    g = nx.parse_adjlist(adj_list)

    g = nx.convert_node_labels_to_integers(g)

    for i in range(n + m):
        if i < n:
            g.nodes[i]['core'] = 1
        else:
            g.nodes[i]['core'] = 0

    return g


def two_opposing_groups(n1=10, n2=10, p1=0.5, p2=0.5, p12=0.5):
    g1 = nx.gnp_random_graph(n=n1, p=p1)
    g2 = nx.gnp_random_graph(n=n2, p=p2)
    g = nx.disjoint_union(g1, g2)

    node1_list = range(n1)
    node2_list = range(n1, n1+n2)

    # assign group ID
    for i in node1_list:
        g.nodes[i]['group'] = 1

    for i in node2_list:
        g.nodes[i]['group'] = 2

    for u, v in g.edges:  # let all edges within groups have correlation 1
        g.edges[u, v]['correlation'] = 1

    # now randomly connect nodes between groups and assign an edge correlation of -1
    for u in node1_list:
        for v in node2_list:
            if np.random.random() < p12:
                g.add_edge(u, v)
                g.edges[u, v]['correlation'] = -1

    return g
    
    
def unbalanced_triangle():
    g = nx.complete_graph(3)
    for u,v in g.edges:
        g.edges[u,v]['correlation'] = 1
    g.edges[0,1]['correlation'] = -1
    return g
    

def unbalanced_ring(n=3, only_one_negative=True):
    nodes = range(n)
    g = nx.Graph()
    
    if only_one_negative:
        for i in range(n):
            node1 = i
            if i == n-1:
                node2 = 0
            else:
                node2 = i+1
                
            g.add_edge(node1, node2)
            if node2 == 0:
                g.edges[node1, node2]['correlation'] = -1
            else:
                g.edges[node1, node2]['correlation'] = 1
            
    else:
        even = True
        for i in range(n):
            node1 = i
            if i == n-1:
                node2 = 0
            else:
                node2 = i+1
                
            g.add_edge(node1, node2)
            if even:
                g.edges[node1, node2]['correlation'] = 1
                even = False
            else:
                g.edges[node1, node2]['correlation'] = -1
                even = True
            
    # enemy_list = [(u,v) for u,v,weight in g.edges.data('correlation') if weight < 0]
    # if len(enemy_list) % 2 == 0:
    #     random_edge = enemy_list[np.random.randint(1, len(enemy_list))]
    #     g.edges[random_edge]['correlation'] = 1
    
    return g
    
    
def ring_graph(n=3, negative_edges=1):
    g = nx.Graph()
    
    for i in range(n):
        node1 = i
        if i == n-1:
            node2 = 0
        else:
            node2 = i+1
            
        g.add_edge(node1, node2)
        
    if negative_edges > 0:
        edges = np.random.permutation([edge for edge in g.edges()])
        for count, edge in enumerate(edges):
            if count < negative_edges:
                g.edges[edge]['correlation'] = -1
            else:
                g.edges[edge]['correlation'] = 1
            
    return g


    
def visualise_graph(g, color_edges=False):
    # layout = nx.spring_layout(g, seed=100)
    layout = nx.spring_layout(g)
    fig = plt.figure(figsize=(4,3.5))
    ax = fig.add_axes((0,0,1,1))
    ax.set_axis_off()
    # fig, ax = plt.subplots(figsize=(4,3.4))

    if color_edges:
        friend_edges = [(x[0],x[1]) for x in g.edges.data('correlation') if x[2] == 1]
        enemy_edges = [(x[0],x[1]) for x in g.edges.data('correlation') if x[2] == -1]
        node_size = 450
        font_size = int(node_size/300 * 12)
        nx.draw_networkx_nodes(g, layout, node_color='white', edgecolors='k', ax=ax, node_size=node_size)
        nx.draw_networkx_labels(g, layout, font_size=font_size)
        nx.draw_networkx_edges(g, layout, edgelist=friend_edges, width=2, edge_color='b', ax=ax, node_size=node_size)
        nx.draw_networkx_edges(g, layout, edgelist=enemy_edges, width=2, edge_color='r', style='dashed', ax=ax, node_size=node_size)
        
        # edge_data = [x[2] for x in g.edges.data('correlation')]
        # nx.draw(g, pos=layout, ax=ax, cmap='Blues', with_labels=True, node_color='white', edgecolors='k',
        #         edge_color=edge_data, edge_vmin=-1, edge_vmax=1, edge_cmap=plt.cm.get_cmap('bwr_r'))
        # cbar = plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(-1, 1), cmap='bwr_r'), ax=[ax], location='left')
        # cbar.set_label('Edges: weight')
    else:
        nx.draw(g, pos=layout, ax=ax, with_labels=True, node_color='white', edgecolors='k')
        
    plt.show()
    plt.close()
    

def test_graph_balance(g, verbose=False, complete_graph=False):
    cycle_list = nx.simple_cycles(nx.DiGraph(g))
    cycle_list = (i for i in cycle_list if len(i)>2)
    if complete_graph:
        # if complete, sufficient to test triads only.
        cycle_list = (i for i in cycle_list if len(i) == 3)
    num_weak = 0
    num_unbalanced = 0
    for i in cycle_list:
        edge_list = group_list_elements_into_tuples(i)
        current_edge_sign = 1
        num_negatives = 0
        for edge in edge_list:
            if g.edges[edge]['correlation'] < 0:
                current_edge_sign *= -1
                num_negatives += 1
        if current_edge_sign < 0:
            if num_negatives == 1:
                num_unbalanced += 1
                if verbose:
                    print(f'{i} cycle is unbalanced')
                else:
                    break
                
            else:
                num_weak += 1
                if verbose:
                    print(f'{i} cycle is weakly balanced')
                    
    if verbose:
        print(f'Unbalanced: {num_unbalanced}')
        print(f'Weak: {num_weak}')
    
    if num_unbalanced:
        #print('The graph is unbalanced')
        return -1
    elif num_weak:
        #print('The graph is weakly balanced')
        return 0
    else:
        #print('The grpah is stongly balanced')
        return 1
    # visualise_graph(g, True)
    

def BA_balance():
    # g = recreate_graph('output/BAfriends_and_enemies')
    # # turn into balanced
    # for edge in g.edges():
    #     g.edges[edge]['correlation'] *= -1
    # g.edges[5,9]['correlation'] *= -1
    # g.edges[2,4]['correlation'] *= -1
    # g.edges[4,6]['correlation'] *= -1
    # g.edges[2,7]['correlation'] *= -1
    # g.edges[7,8]['correlation'] *= -1
    # 
    # g.edges[2,4]['correlation'] *= -1  # turn into unbalanced
    # g.edges[3,5]['correlation'] *= -1  # turn into weak
    
    g = ring_graph(4, 4)
    g.edges[2,3]['correlation'] = 1
    
    visualise_graph(g, True)
    
    

if __name__ == '__main__':
    # BA_balance()
    print(np.full(3,False))
    
    


