import numpy as np
from graph_tool import Graph
from graph_tool.topology import shortest_distance

def dwell_time_all_agents(asymptotic: list[int]) -> tuple[list[int], list[int], list[int]]:
    steps = np.array(asymptotic)
    # last index at dwell
    dwell_indices = np.where(np.diff(steps) < 0)[0].tolist()
    if steps[-1] > 0:
        dwell_indices.append(len(asymptotic) - 1)
    dwell_times = [int(asymptotic[i]) for i in dwell_indices]
    return dwell_indices, dwell_times

def dwell_time_per_agent_with_means(asymptotic: list[list[int]], mean_list) \
    -> list[list[tuple[int, float]]]:
    """
    for each agent, give a list of its dwell times (time, mean bias at that time)
    """
    dwells = [dwell_time_all_agents(agent) for agent in asymptotic]
    return [list(zip(
            times,
            # indices used only to query the biases at the time of the dwell
            mean_list[indices, agent].tolist() 
        ))
    for agent, (indices, times) in enumerate(dwells)]

def dwell_time_per_agent(asymptotic: list[list[int]]) -> list[list[int]]:
    """input: asymptotic - per agent, list of number of steps spent in asymptotic learning at time t
    output: dwell time - per agent, all maxima of asymptotic
    """
    return [dwell_time_all_agents(agent)[1] for agent in asymptotic]

def dwell_time_per_agent_by_distance_to_partisan(
    graph: Graph, 
    num_partisans: int, 
    agent_is_asymptotic: list[list[int]],
    mean_list = None):

    # NOTE: the following assumes we do not find shortest path based on negative edge weights
    # this needs to change otherwise!
    # for i, u in zip(range(num_partisans), graph.iter_vertices()):
    #     for _, v in zip(range(i + 1, num_partisans), graph.iter_vertices()):
    # connect all partisans with each other, 0 cost edge
    for u in range (num_partisans):
        for v in range (u+1, num_partisans):
            e = graph.add_edge(u, v)
            graph.ep.friendliness[e] = 0

    dist = shortest_distance(graph, source=0, weights=graph.ep.friendliness)
    dist_per_node = dist.a.astype(int).tolist() # type: list[int]
    
    if mean_list is None:
        dwells = dwell_time_per_agent(agent_is_asymptotic)
    else:
        dwells = dwell_time_per_agent_with_means(agent_is_asymptotic, mean_list)

    max_dist = max(dist_per_node)
    dwells_by_distance: list[list[tuple[int, float] | int]] = [[] for _ in range(max_dist + 1)]
    number_of_agents_by_distance = [0 for _ in range(max_dist + 1)]
    for dist, dwell in zip(dist_per_node, dwells):
        dwells_by_distance[dist].extend(dwell)
        number_of_agents_by_distance[dist] += 1
    
    parts = zip(number_of_agents_by_distance, dwells_by_distance)
    return list(parts)

    # Agverage each sim
    # res = []
    # for n, d in parts:
    #     m = mean(d)
    #     s = stdev(d, m) if n > 1 else 0
    #     res.append((n, m, s))
    # return res

def asymptotic_per_agent(dists):
    steps = len(dists) - 1
    n = len(dists[0])
    asymptotic = np.zeros((n, steps), dtype=int)
    row = np.zeros(n, dtype=int)
    for i, (prior, posterior) in enumerate(zip(dists, dists[1:])):
        largest_change = np.max(np.abs(posterior - prior), axis=1)
        largest_peak = 0.01 * np.max(prior, axis=1)
        agent_is_asymptotic = largest_change < largest_peak  # one element for each agent
        row = np.where(agent_is_asymptotic, row + 1, 0)
        asymptotic[:, i] = row
        prior = posterior
    return asymptotic
