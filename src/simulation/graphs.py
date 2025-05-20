from random import choice
import graph_tool.all as gt
from numpy.random import uniform
from typing import Callable
import numpy as np

edge_generator_type = Callable[[], int]

def get_edge_generator(edges: str) -> edge_generator_type:
    if edges == "friends" or edges == "allies":
        return lambda: ADJ_FRIEND
    elif edges == "enemies" or edges == "opponents":
        return lambda: ADJ_ENEMY
    elif edges == "random":
        return gen_relationship_binary
    elif edges == "random_unif":
        return gen_relationship_uniform
    
    raise NotImplementedError("Invalid edge type")

def show_graph(g, edge_pen_width = 1, edge_property_name = 'friendliness'):
    from graph_tool.draw import graph_draw

    # Define colors
    grey_color = (128/255, 128/255, 128/255, 1) # grey
    black_color = (0, 0, 0, 1) # black
    white_color = (1, 1, 1, 1) # white
    transparent_color = (0, 0, 0, 0) # transparent

    # Get graph and vertex count
    n_vertices = g.num_vertices()

    # Set vertex fill color and outline color based on index
    vertex_fill_color = g.new_vertex_property('vector<double>')
    # vertex_fill_color = transparent_color
    vertex_outline_color = g.new_vertex_property('vector<double>')
    for i in range(n_vertices):
        vertex_fill_color[g.vertex(i)] = white_color
        vertex_outline_color[g.vertex(i)] = black_color

    # Draw the graph
    edge_color_map = {-1.0: (199/255, 27/255, 0, 1),  # red
                    1.0: (0, 128/255, 255/255, 1),  # blue
                    0.0: (0, 0, 0, 0)}  # black
    edge_color = g.new_edge_property('vector<double>')
    for f, e in zip(g.ep[edge_property_name], g.edges()):
        edge_color[e] = edge_color_map[f]

    graph_draw(g, vertex_text=g.vertex_index, vertex_fill_color=vertex_fill_color,
            # vertex_halo = True,
            vertex_color=vertex_outline_color, 
            vertex_text_color = black_color, edge_color=edge_color,edge_pen_width=edge_pen_width)




def friendliness_mat(g):
    # takes roughly 300ms for graph of n=10
    return gt.adjacency(g, weight=g.ep.friendliness).toarray()


def adjacency_mat(g):
    return gt.adjacency(g).toarray()


def graph_shuffle_indices(g: gt.Graph):
    verts = np.arange(g.num_vertices())
    # rng = np.random.default_rng(seed=42)
    # rng.shuffle(verts)
    np.random.shuffle(verts)
    A = adjacency_mat(g)
    A = A[verts][:, verts]  # shuffle the rows and columns

    return graph_from_friendliness_mat(A)

# def graph_from_friendliness_mat(A):
#     g = create_model_graph()
#     A = np.tril(A) # only add each edge once, assume an undirected graph
#     g.add_vertex(A.shape[0] - g.num_vertices())
#     edges = np.transpose(np.transpose(A).nonzero())
#     g.add_edge_list(edges)
#     g.ep.friendliness.a = A[A.nonzero()]

#     # assert(np.all(friendliness_mat(graph_from_friendliness_mat(A)) == A))
#     # assert(np.all(adjacency_mat(graph_from_friendliness_mat(A)) == A))
#     return g



def graph_from_friendliness_mat(A):
    g = create_model_graph()
    num_vertices = A.shape[0]
    
    # Add vertices to the graph
    g.add_vertex(num_vertices)
    
    # Extract lower triangular part to avoid duplicate edges
    A_lower = np.tril(A)
    
    # Get the non-zero entries from the lower triangular matrix
    rows, cols = A_lower.nonzero()
    edges = list(zip(rows, cols))
    
    # Add edges to the graph
    for edge in edges:
        g.add_edge(edge[0], edge[1])
    
    # Add the weights to the edges
    weights = g.new_edge_property("double")
    for edge in edges:
        weights[g.edge(edge[0], edge[1])] = A_lower[edge[0], edge[1]]
    
    g.ep.friendliness = weights

    return g


def create_model_graph(seed_graph: gt.Graph=None) -> gt.Graph:
    g = seed_graph or gt.Graph(directed=False)
    # nodes represent people
    # edges represent 'knowing' relationships of people

    # friendliness:  # or g.ep.friendliness
    #   -1 if people are enemies
    #   +1 if people are friends'
    g.edge_properties["friendliness"] = g.new_edge_property("double")

    # each person has a prior distr, mean and std
    # or g.vp.prior_mean and g.vp.prior_sd
    g.vertex_properties["prior_mean"] = g.new_vertex_property("double")
    g.vertex_properties["prior_sd"] = g.new_vertex_property("double")
    g.vertex_properties["prior_distr"] = g.new_vertex_property(
        "vector<double>")

    # g.graph_properties['step'] = g.new_graph_property('int')

    return g

ADJ_FRIEND = 1
ADJ_ENEMY = -1


def add_relationship(g, v1, v2, friendliness):
    e = g.add_edge(v1, v2)  # they know each other

    g.ep.friendliness[e] = friendliness  # if they like each other


def add_friends(g, v1, v2):
    add_relationship(g, v1, v2, ADJ_FRIEND)


def add_enemies(g, v1, v2):
    add_relationship(g, v1, v2, ADJ_ENEMY)


def pair_of_allies():
    g = create_model_graph()
    v1, v2 = g.add_vertex(2)
    add_friends(g, v1, v2)
    return g


def pair_of_opponents():
    g = create_model_graph()
    v1, v2 = g.add_vertex(2)
    add_enemies(g, v1, v2)
    return g

def gen_complete_graph(n: int, edge_generator: edge_generator_type) -> gt.Graph:
    g = create_model_graph()
    v = g.add_vertex()
    vlist = [v]

    for _i in range(1, n):
        u = g.add_vertex()
        for v in vlist:
            add_relationship(g, u, v, edge_generator())
        vlist.append(u)

    return g

def gen_triad(num_enemies: int) -> gt.Graph:
    g = create_model_graph()
    v1, v2, v3 = g.add_vertex(3)
    if num_enemies == 0:
        add_friends(g, v1, v2)
        add_friends(g, v1, v3)
        add_friends(g, v2, v3)
    elif num_enemies == 1:
        add_friends(g, v1, v2)
        add_enemies(g, v1, v3)
        add_friends(g, v2, v3)
    elif num_enemies == 2:
        add_enemies(g, v1, v2)
        add_friends(g, v2, v3)
        add_enemies(g, v1, v3)
    elif num_enemies == 3:
        add_enemies(g, v1, v2)
        add_enemies(g, v1, v3)
        add_enemies(g, v2, v3)
    else:
        raise ValueError("Invalid number of enemies")
    return g


# def gen_incomplete_link(edge_list: list) -> gt.Graph:
#     if len(edge_list) != 2:
#         raise ValueError("Invalid number of vertices")
#     g = create_model_graph()
#     v1,v2,v3 = g.add_vertex(3)
#     if edge_list[0] == 1:
#         add_friends(g, v1, v2)
#     elif edge_list[0] == -1:
#         add_enemies(g, v1, v2)
#     if edge_list[1] == 1:
#         add_friends(g, v2, v3)
#     elif edge_list[1] == -1:
#         add_enemies(g, v2, v3)
#     else:
#         raise ValueError("Invalid edge type")
#     return g


# More general version
def gen_incomplete_link(edge_list: list) -> gt.Graph:
    num_vertices = len(edge_list) + 1
    g = create_model_graph()
    vertices =  list(g.add_vertex(num_vertices))
    for v, edge in zip(range(num_vertices-1), edge_list):
        add_relationship(g, v, v + 1, edge)
    return g


def gen_bba_graph(n: int, m: int, edge_generator: edge_generator_type) -> gt.Graph:
    """Generates a Barabási-Albert network"""
    g = gt.price_network(n, m, directed=False)

    g = create_model_graph(g)
    g = graph_shuffle_indices(g)

    import numpy as np
    g.ep.friendliness.a = np.array([edge_generator() for x in range(g.num_edges())])
    # g.ep.friendliness.a = np.fromfunction(edge_generator, (g.num_edges(),), dtype=float)
    return g


def gen_bba_graph_mixed(n: int, m: int) -> gt.Graph:
    """Generates a Barabási-Albert network"""
    g = gt.price_network(n, m, directed=False)

    g = create_model_graph(g)
    g = graph_shuffle_indices(g)

    import numpy as np
    g.ep.friendliness.a = np.random.choice((1, -1), g.num_edges())
    # g.ep.friendliness.a = np.fromfunction(edge_generator, (g.num_edges(),), dtype=float)
    return g

def gen_relationship_binary():
    return choice([ADJ_ENEMY, ADJ_FRIEND])

# this will return float in range [-1, 1]
def gen_relationship_uniform():
    return uniform(ADJ_ENEMY, ADJ_FRIEND)


def complete_graph_of_friends(n):
    return gen_complete_graph(n, lambda: ADJ_FRIEND)


def complete_graph_of_enemies(n):
    return gen_complete_graph(n, lambda: ADJ_ENEMY)

gen_complete_graph_of_allies = complete_graph_of_friends
gen_complete_graph_of_opponents = complete_graph_of_enemies

def complete_graph_of_random(n):
    return gen_complete_graph(n, gen_relationship_binary)


def complete_graph_of_random_uniform(n):
    return gen_complete_graph(n, gen_relationship_uniform)

Graph = gt.Graph

class GraphGenerators:

    @staticmethod
    def pair_of_allies() -> Graph:
        return pair_of_allies()
    
    @staticmethod
    def pair_of_opponents() -> Graph:
        return pair_of_opponents()
    
    @staticmethod
    def triad(num_enemies: int = 0) -> Graph:
        return gen_triad(num_enemies)
    
    @staticmethod
    def unbalanced_triad() -> Graph:
        return gen_triad(num_enemies=1)
    
    @staticmethod
    def balanced_triad() -> Graph:
        return gen_triad(num_enemies=2)
    
    @staticmethod
    def incomplete_link(edge_list: list) -> Graph:
        return gen_incomplete_link(edge_list)
   
    @staticmethod
    def complete_graph_of_allies(n: int) -> Graph:
        return complete_graph_of_friends(n)
    
    @staticmethod
    def complete_graph_of_opponents(n: int) -> Graph:
        return complete_graph_of_enemies(n)

    @staticmethod
    def complete_graph_of_mixed(n: int) -> Graph:
        """Generates a complete graph of agents with mixed allegiances (allies and opponents)"""
        return complete_graph_of_random(n)

    @staticmethod
    def complete_graph_of_random(n: int) -> Graph:
        """Generates a complete graph of agents with random allegiances (allies and opponents)"""
        return complete_graph_of_random(n)
    
    @staticmethod
    def complete_graph_of_random_uniform(n: int) -> Graph:
        """Generates a complete graph of agents with random uniform allegiances (-1 to 1 uniform)"""
        return complete_graph_of_random_uniform(n)
    
    @staticmethod
    def ba_graph(n: int, m: int, edge_generator: edge_generator_type) -> Graph:
        """Generates a Barabási-Albert graph with n nodes and m out-degree for each node, using edge_generator to specify allegiances"""
        return gen_bba_graph(n, m, edge_generator)

    @staticmethod
    def ba_graph_of_mixed(n: int, m: int) -> Graph:
        """Generates a Barabási-Albert graph of mixed allegiances of allies an opponents with n nodes and m out-degree for each node, using random choice for allegiances"""
        return gen_bba_graph_mixed(n, m)

    @staticmethod
    def ba_graph_of_allies(n: int, m: int) -> Graph:
        """Generates a Barabási-Albert graph of allies with n nodes and m out-degree for each node"""
        return gen_bba_graph(n, m, lambda: ADJ_FRIEND)
    
    @staticmethod
    def ba_graph_of_opponents(n: int, m: int) -> Graph:
        """Generates a Barabási-Albert graph of opponents with n nodes and m out-degree for each node"""
        return gen_bba_graph(n, m, lambda: ADJ_ENEMY)


def add_ally_with_all(g: gt.Graph):
    """Add a new vertex, and make it a friend with all students"""
    new_v = g.add_vertex()
    for v2 in g.vertices():
        if v2 != new_v:
            add_friends(g, v2, new_v)
    return g, new_v
