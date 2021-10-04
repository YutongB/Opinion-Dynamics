import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from read_graph import recreate_graph
import itertools
import timeit
from PIL import Image


"""
Intended to store various fnctions related to generating networks
I actually didn't even use any of these graph generating functions for the paper

Somehow it also includes basic visualisation from visualise_graph
"""


# convinience function for testing cycles
def group_list_elements_into_tuples(l, size=2, cycle_back=True):
    """
    if l = [0,1,2,3]
    then returns [[0,1], [1,2], [2,3], [3,0]] (with default params)
    """
    
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
    
    Assumes that all cores are connected to each other and all peripheries are not connected to each other
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
        g.nodes[i]['group'] = 0

    for i in node2_list:
        g.nodes[i]['group'] = 1

    for u, v in g.edges:  # let all edges within groups have correlation 1
        g.edges[u, v]['correlation'] = 1

    # now randomly connect nodes between groups and assign an edge correlation of -1
    for u in node1_list:
        for v in node2_list:
            if np.random.random() < p12:
                g.add_edge(u, v)
                g.edges[u, v]['correlation'] = -1

    return g
    

def complete_opposing_groups(num_nodes=10, num_groups=2):
    """
    num_nodes: can be int of list of ints
               int: the total number of nodes the graph has
               list: each element corresponds to the number of nodes each group has
    
    num_groups: int
                if num_nodes is int, this determines how many groups to evenly split the nodes into. Otherwise, it is ignored.
    """
    if isinstance(num_nodes, int): 
        if num_groups > num_nodes:
            raise ValueError(f'num_groups ({num_groups}) cannot exceed num_nodes ({num_nodes})')
        # convert num_nodes into the list equivalent
        total_nodes = num_nodes
        nodes_per_group = num_nodes // num_groups
        remainder = num_nodes % num_groups
        num_nodes = [nodes_per_group for i in range(num_groups)]
        for i in range(remainder):
            num_nodes[i] += 1
    else:
        total_nodes = sum(num_nodes)
    
    g = nx.complete_graph(total_nodes)
    current_group_label = 0
    count = 0
    for i in range(total_nodes):
        g.nodes[i]['group'] = current_group_label
        count += 1
        if count >= num_nodes[current_group_label]:
            count = 0
            current_group_label += 1
    
    for u,v in g.edges():
        if g.nodes[u]['group'] == g.nodes[v]['group']:
            g.edges[u,v]['correlation'] = 1
        else:
            g.edges[u,v]['correlation'] = -1
            
    visualise_graph(g, True)
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

"""
Graph visualisation
"""

    
def visualise_graph(g, color_edges=False, layout=None, save_file=''):
    if layout is None:
        # seed for consistent layout
        layout = nx.spring_layout(g, seed=100)
    fig = plt.figure(figsize=(4,3.5))
    #fig = plt.figure()
    #fig = plt.figure(figsize=(3,3))
    ax = fig.add_axes((0,0,1,1))
    ax.set_axis_off()
    
    labels = {}
    for i in range(g.number_of_nodes()):
        labels[i] = str(i+1)

    if color_edges:
        friend_edges = [(x[0],x[1]) for x in g.edges.data('correlation') if x[2] == 1]
        enemy_edges = [(x[0],x[1]) for x in g.edges.data('correlation') if x[2] == -1]
        node_size = 450
        font_size = int(node_size/300 * 11)
        nx.draw_networkx_nodes(g, layout, node_color='white', edgecolors='k', ax=ax, node_size=node_size)
        nx.draw_networkx_labels(g, layout, font_size=font_size, ax=ax, labels=labels)
        nx.draw_networkx_edges(g, layout, edgelist=friend_edges, width=2, edge_color='b', ax=ax, node_size=node_size)
        nx.draw_networkx_edges(g, layout, edgelist=enemy_edges, width=2, edge_color='r', style='dashed', ax=ax, node_size=node_size)
    else:
        d = dict(g.degree)
        #nx.draw(g, nodelist=d.keys(), node_size=[v * 10 for v in d.values()])
        
        nx.draw(g, pos=layout, ax=ax, with_labels=True, node_color='white', edgecolors='k', node_size=1000, font_size=17, labels=labels)
        
    # ax.set_xlim(-0.6, 0.6)
    # ax.set_ylim(-0.6, 0.6)
    if save_file:
        fig.savefig(save_file)
    else:
        plt.show()
    plt.close()
    

def test_graph_balance(g, verbose=False, complete_graph=False):
    """	
    Warning: very slow for big graphs, due to huge number of cycles!
    
    Return values:	
    -1: Unbalanced	
    0 : Weakly balanced	
    1 : Strongly balanced	
    """
    if complete_graph:
        # if complete, sufficient to test triads only.
        cycle_list = itertools.combinations(range(g.number_of_nodes()), 3)
    else:
        cycle_list = nx.simple_cycles(nx.DiGraph(g))
        cycle_list = (i for i in cycle_list if len(i)>2)
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
    

def find_frac_of_unbalanced_cycles(g):
    # NOTE: considers weak and unbalanced to be the same
    cycle_list = nx.simple_cycles(nx.DiGraph(g))
    cycle_list = (i for i in cycle_list if len(i)>2)
    
    num_unbalanced = 0
    for count, i in enumerate(cycle_list):
        edge_list = group_list_elements_into_tuples(i)
        current_edge_sign = 1
        for edge in edge_list:
            current_edge_sign *= g.edges[edge]['correlation']
        if current_edge_sign < 0:
            num_unbalanced += 1
    
    return num_unbalanced / (count + 1)
    

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


def get_circular_layout(start_theta=np.pi/2, num_points=3):
    def point_on_circle(theta):
        return 0.5* np.cos(theta), 0.5* np.sin(theta)
        
    theta_list = np.linspace(start_theta, start_theta + 2* np.pi, num_points+1)[:-1]
    return dict(zip(range(num_points), (point_on_circle(i) for i in theta_list)))
    

def draw_triads():
    g_list = [nx.complete_graph(3) for i in range(4)]
    possible_edge_weight_combo = list(itertools.combinations_with_replacement([1,-1], 3))
    possible_edge_combo = list(itertools.combinations(list(range(3)), 2))
    for i in range(len(possible_edge_weight_combo)):
        for j in range(len(possible_edge_combo)):
            g_list[i].edges[possible_edge_combo[j]]['correlation'] = possible_edge_weight_combo[i][j]
    
    for edge in g_list[1].edges():
        g_list[1].edges[edge]['correlation'] = 1
    g_list[1].edges[0,1]['correlation'] = -1
    
    file_list = [f'tri{count}.png' for count in range(len(g_list))]
    for count, g in enumerate(g_list):
        visualise_graph(g, True, get_circular_layout(np.pi/2), file_list[count])

    for f in file_list:
        im = Image.open(f)
        im = im.crop((0, 0, 200, 160))
        im.save(f'{f}')
        
        

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 17})
    g = nx.Graph()
    for i in range(4):
        g.add_node(i)
        
    g.add_edge(0,1)
    g.add_edge(2,1)
    g.add_edge(3,1)
    g.add_edge(2,3)
    visualise_graph(g, layout=get_circular_layout(num_points=4))