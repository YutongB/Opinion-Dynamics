from random import choice
import graph_tool.all as gt
from numpy.random import uniform
from typing import Callable

edge_generator_type = Callable[[], int]

def get_edge_generator(edges: str) -> edge_generator_type:
    if edges == "friends" or edges == "allies":
        return lambda: ADJ_FRIEND
    elif edges == "enemies":
        return lambda: ADJ_ENEMY
    elif edges == "random":
        return gen_relationship_binary
    elif edges == "random_unif":
        return gen_relationship_uniform
    
    raise NotImplementedError("Invalid edge type")

def draw_graph(g, show_vertex_labels=False, width=150):

    # edge_color_map = {-1: (1, 0, 0, 1), 1: (0, 1, 0, 1), 0: (0, 0, 0, 0)}
    # edge_colors = [edge_color_map[int(e)] for e in g.ep.friendliness]

    gt.draw.graphviz_draw(g,
                  vertex_text=g.vertex_index if show_vertex_labels else None,
                  #   edge_color=edge_colors,
                  edge_text=g.ep.friendliness,
                  output_size=(width, width))

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

    import numpy as np
    g.ep.friendliness.a = np.array([edge_generator()] * g.num_edges())
    # g.ep.friendliness.a = np.fromfunction(edge_generator, (g.num_edges(),), dtype=float)
    return g


def gen_bba_graph_mixed(n: int, m: int) -> gt.Graph:
    """Generates a Barabási-Albert network"""
    g = create_model_graph()
    g = gt.price_network(n, m, directed=False)

    g = create_model_graph(g)

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


def complete_graph_of_random(n):
    return gen_complete_graph(n, gen_relationship_binary)


def complete_graph_of_random_uniform(n):
    return gen_complete_graph(n, gen_relationship_uniform)
