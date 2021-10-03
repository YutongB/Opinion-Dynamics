import graph_tool.all as gt
import numpy as np
from sim import *

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
    cluster_ids.a = np.full(g.num_vertices, -1)
    gt.dfs_search(g, visitor=ClusterVisitor(cluster_ids, g.ep.friendliness))
    return cluster_ids.a

class UnbalancedVisitor(gt.DFSVisitor):
    """
    Detect unbalanced graph
    """
    def __init__(self, cluster_ids, friendliness):
        self.cluster_ids = cluster_ids
        self.friendliness = friendliness

    def examine_edge(self, e):
        u, v = e.source(), e.target()
        # unbalanced: impossible to group agents into a cluster
        # therefore, we'd have two agents in the same cluster that _are_ opponents
        if self.friendliness[e] < 0:  # opponents
            if self.cluster_ids[u] == self.cluster_ids[v]:
                raise gt.StopSearch()  # unbalanced!

from enum import Enum
class Balance(Enum):
    UNBALANCED = 0
    WEAKLY = 1
    STRONGLY = 2

def test_graph_balance(g):
    cluster_ids = get_clusters(g)

    # strongly balanced: every agent can be grouped into one or two distinct clusters
    # weakly balanced: every agent can be grouped into more than two distinct clusters
    # unbalanced: impossible to group agents into a cluster

    try:
        gt.dfs_search(g, visitor=UnbalancedVisitor(cluster_ids, g.ep.friendliness))
    except gt.StopSearch:
        return Balance.UNBALANCED

    num_clusters = np.unique(cluster_ids)
    if num_clusters <= 2:
        return Balance.STRONGLY
    else:
        return Balance.WEAKLY

def gen_balanced(k, n=1e4, m=3):
    """
    k: number of clusters
    n: number of nodes
    m: number of edges to add per new node added
    """

    # generates an undirected Barabási-Albert network
    g = create_model_graph(gt.price_network(N=n, m=m, directed=False))

    # initially all friendly
    g.ep.friendliness.a[:] = 1

    # 

    
    return g





