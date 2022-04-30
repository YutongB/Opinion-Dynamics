from enum import IntEnum
import graph_tool.all as gt
import numpy as np
from src.simulation.sim import *
from src.simulation.graphs import create_model_graph


class ClusterVisitor(gt.DFSVisitor):
    """
    Does a DFS traversal of the graph O(V + E) to make cluster ids

    use number of unique values in cluster to check strongly or weakly balanced
    """

    def __init__(self, cluster_ids, friendliness):
        self.cluster_ids = cluster_ids
        self.friendliness = friendliness

    def discover_vertex(self, u):
        self.cluster_ids[u] = u

    def examine_edge(self, e):
        if self.friendliness[e] >= 0:  # allies
            s = self.cluster_ids[e.source()]
            t = self.cluster_ids[e.target()]
            if t == -1 or t == s:
                self.cluster_ids[e.target()] = s
            else:
                # already given a cluster number, let's replace
                # each with this cluster number with the cluster number we just ran into.
                self.cluster_ids.a[self.cluster_ids.a == s] = t


def get_clusters(g):
    """
    Gets the clusters of a graph (graph-tool), returned as an array of cluster ids for each vertex

    cluster := nonempty set of agents such that:
    - any two agents in same cluster are not opponents => all edge weights >= 0, and
    - any two agents from different clusters are not allies.  

    worst case, E = O(V^2)  (complete graph); usual case, E ~ O(V)
    """
    cluster_ids = g.new_vertex_property('int')
    cluster_ids.a = np.full(g.num_vertices(), -1)
    gt.dfs_search(g, visitor=ClusterVisitor(cluster_ids, g.ep.friendliness))
    return cluster_ids


class Balance(IntEnum):
    UNBALANCED = 0
    WEAKLY = 1
    STRONGLY = 2


balance_title = {Balance.UNBALANCED: "Unbalanced",
                 Balance.WEAKLY: "Weakly",
                 Balance.STRONGLY: "Strongly"}


def test_graph_balance(g):
    cluster_ids = get_clusters(g)

    # strongly balanced: every agent can be grouped into one or two distinct clusters
    # weakly balanced: every agent can be grouped into more than two distinct clusters
    # unbalanced: impossible to group agents into a cluster

    for e in gt.dfs_iterator(g):
        u, v = e.source(), e.target()
        # unbalanced: impossible to group agents into a cluster
        # therefore, we'd have two agents in the same cluster that _are_ opponents
        if g.ep.friendliness[e] < 0:  # opponents
            if cluster_ids[u] == cluster_ids[v]:
                return Balance.UNBALANCED

    cluster_ids = cluster_ids.a
    num_clusters = len(np.unique(cluster_ids))
    if num_clusters <= 2:
        return Balance.STRONGLY
    else:
        return Balance.WEAKLY


def gen_balanced(k, n=1e4, m=3, verbose=False):
    """
    k: number of clusters
    n: number of nodes
    m: number of edges to add per new node added
    """

    # generates an undirected Barabási-Albert network
    g = create_model_graph(gt.price_network(N=n, m=m, directed=False))
    g.vp.cluster_id = g.new_vp("int")

    if verbose:
        print("running gen_balanced with k=", k)

    # TODO: There's a bug below, try:
    # np.unique(get_clusters(gen_balanced(k=2, n=1000, m=3)))

    cluster_sizes = np.random.multinomial(n-k, np.ones(k)/k) + 1
    nodes_id = np.repeat(np.arange(k), cluster_sizes)
    np.random.shuffle(nodes_id)
    g.vp.cluster_id.a[:] = nodes_id

    is_same_group = nodes_id.reshape(-1, 1) == nodes_id

    edge_values = is_same_group * 2 - 1   # False -> -1, True -> 1

    edges = g.get_edges()
    g.ep.friendliness.a = edge_values[edges[:, 0], edges[:, 1]]

    return g


def gen_unbalanced(p=0.5, n=1e4, m=3):
    """
    p = probability of changing to -1  (being an opponent edge)
    """
    # NOTE: technically (rarely) can produce balanced graphs.

    g = create_model_graph(gt.price_network(N=n, m=m, directed=False))
    M = g.ep.friendliness.a.shape[0]

    g.ep.friendliness.a = np.random.choice(a=[-1, 1], size=M, p=[p, 1-p])
    return g


def gen_strongly_balanced(**kwargs):
    k = 1 if np.random.random() < 0.5 else 2
    return gen_balanced(k=k, **kwargs)


def gen_weakly_balanced(p=0.5, **kwargs):
    k = 3 + np.random.geometric(p=p)
    return gen_balanced(k=k, **kwargs)


def gen_balanced_type(type, **kwargs):
    if type == Balance.WEAKLY:
        return gen_weakly_balanced(**kwargs)
    elif type == Balance.STRONGLY:
        return gen_strongly_balanced(**kwargs)
    else:
        return gen_unbalanced(**kwargs)


def balanced_graph_gen(type, n, m, test_balance=False):
    def graph_generator():
        while True:
            g = gen_balanced_type(type, n=n, m=m)

            # for debug, we assure we're generating stuff of the right type
            if test_balance and (actual_type := test_graph_balance(g)) != type:
                print(
                    f"WARN! Attempted generating {type} but generated {actual_type}. Skipping.")
                continue
            return g
    return graph_generator


def run_balanced(sim_params, threshold=1e3, n=1e4, m=3):
    # return {t: run_balanced_one_type(t, sim_params, threshold=threshold, n=n, m=m) for t in Balance}
    n = int(n)
    runs = int(threshold)

    def run_one_balanced(t):
        gen_graph = balanced_graph_gen(t, n=n, m=int(m))

        results = do_ensemble(runs=runs, gen_graph=gen_graph,
                                          title=balance_title[t],
                                          sim_params=sim_params)
        return [r.steps if r.asymptotic else -1 for r in results if r is not None]

    b = list(iter(Balance))[::-1]  # do strong first
    return {balance_title[t]: run_one_balanced(t) for t in b}


def draw_cluster_graph(g):
    edge_color_map = {-1.0: (1, 0, 0, 1),  # red
                      1.0: (0, 1, 0, 1),  # green
                      0.0: (0, 0, 0, 0)}  # black

    edge_color = g.new_ep('vector<double>')
    for f, e in zip(g.ep.friendliness, g.edges()):
        edge_color[e] = edge_color_map[f]

    try:
        clusters = g.vp.cluster_id
    except KeyError:
        clusters = None

    gt.graph_draw(g, vertex_text=clusters,
                  edge_color=edge_color,
                  output_size=(300, 300))
