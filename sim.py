# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# The import order is important
import matplotlib as mpl
mpl.use('cairo') 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# %%
import graph_tool.all as gt
import numpy as np
from numpy.random import seed, random, randint, uniform
import scipy
from scipy.stats import norm


# %%
def create_model_graph():
    g = gt.Graph(directed=False)
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
    g.vertex_properties["prior_distr"] = g.new_vertex_property("vector<double>")

    return g


# %%
def draw_graph(g, show_vertex_labels=False, width=150):

    edge_color_map = {-1: (1, 0, 0, 1), 1: (0, 1, 0, 1)}
    edge_colors = [edge_color_map[e] for e in g.ep.friendliness]

    gt.graph_draw(g, 
                  vertex_text=g.vertex_index if show_vertex_labels else None,
                 # edge_color=edge_colors,
                  edge_text=g.ep.friendliness, 
                  output_size=(width,width))


# %%
ADJ_FRIEND = 1
ADJ_ENEMY = -1

def add_relationship(g, v1, v2, friendliness):
    e = g.add_edge(v1, v2) # they know each other

    g.ep.friendliness[e] = friendliness  # if they like each other

def add_friends(g, v1, v2):
    add_relationship(g, v1, v2, ADJ_FRIEND)

def add_enemies(g, v1, v2):
    add_relationship(g, v1, v2, ADJ_ENEMY)


# %%
def create_pair():
    """
    Creates graph of 2 nodes and one edge.
    """
    g = create_model_graph()

    v1 = g.add_vertex()
    v2 = g.add_vertex()
    e = g.add_edge(v1, v2)

    g.ep.adj[e] = ADJ_FRIEND

    return g

# non-biased - theta = 0.5


# %%
g = create_model_graph()

v1, v2 = g.add_vertex(2)

add_enemies(g, v1, v2)

draw_graph(g, show_vertex_labels=True)


# %%
seed(42)


# %%
HEADS = 1  # "success" is for the coin to land heads
TAILS = 0
# a Ber(0.5) trial
def toss_coin():
    return randint(0, 2)


# %%
def bias():
    # discretise the continuous variable theta (bias) into 21 regularly-spaced
    # values  (0.00, 0.05, ..., 1.00)
    return np.linspace(0, 1, 21)

def bias_mat(num_priors):
    # generate matrix of num_prior rows and 
    # 21 columns (bias() for each row of the matrix)
    return np.tile(bias(), (num_priors, 1))

def fwhm_to_sd(fwhm):
    # full width half mean
    # 2 * sqrt(2 * ln(2))  (see wiki) (don't calculate sqrt or ln)
    return fwhm / 2.3548200450309493820231386529

def generate_priors(num_priors, mean_range=(0.0, 1.0), fwhm_range=(0.2, 0.8)):
    prior_mean = uniform(*mean_range, num_priors)
    prior_fwhm = uniform(*fwhm_range, num_priors)
    prior_sd = fwhm_to_sd(prior_fwhm)

    return prior_mean, prior_sd

def prior_distr(prior_mean, prior_sd):
    return norm.pdf(bias(), loc=prior_mean, scale=prior_sd)

def prior_distr_vec(num_priors, prior_mean, prior_sd):
    # ensure norm.pdf behaves (broadcasting - scipy)
    prior_mean.shape = prior_sd.shape = (num_priors, 1)
    return norm.pdf(bias_mat(num_priors), loc=prior_mean, scale=prior_sd)


# %%
def run_simulation(g, max_steps=1e4):
    """
    max_steps (T in the paper) - maximum number of steps to run the simulation
    """
    # steps 2-3 of Probabilistic automaton, until t = T
    for t in range(max_steps):
        step_simulation(g)


# %%
def likelihood_fn(state, bias):
    """
    state: coin toss value (TODO generalise)
    bias: θ, the value for the bias
    The likelihood encodes the probability of observing a heads, given θ
    """
    if state == 0:
        return bias
    return 1 - bias


# %%
def coin_toss_likelihood():
    """
    coin toss likelihood - encodes the probability of observing a heads, given bias θ
    only 2 possible outcomes in a coin toss:
    row 0 (tails) : 1 - θ
    row 1 (heads) : θ 
    """
    return np.array((1 - bias(), bias()))


# %%
def normalising_constant_i(likelihood, prior_distr):
    """
    sum_θ{  P(S(T) | θ) * x_i (t, θ)    }
    """
    return np.sum(likelihood * prior_distr, axis=1)


# %%
def normalising_constant(likelihood, graph_prior_distr):
    """
    value i is normalising constant for node / person i
    sum_θ{  P(S(T) | θ) * x_i (t, θ) }
    """
    # for 1 person
    if len(graph_prior_distr.shape) == 1:
        return np.sum(np.matmul(likelihood, graph_prior_distr))

    # ensure the prior distribution has the right shape
    if likelihood.shape[1] == graph_prior_distr.shape[1]:
        graph_prior_distr = graph_prior_distr.T
    return np.sum(np.matmul(likelihood, graph_prior_distr), axis=1)


# %%
def debug_state(g):
    verts = g.iter_vertices([g.vp.prior_mean, g.vp.prior_sd, g.vp.prior_distr])
    for v, prior_mean, prior_sd, prior_distr in verts:
        print(v, prior_mean, prior_sd, prior_distr)

    plot_distr(g)

def plot_distr(distr, title=None):
    fig, ax = plt.subplots()

    for v, d in enumerate(distr):
        ax.plot(bias(), d, label=str(v))

    ax.set_xlabel(r'Bias, $\theta$')
    ax.set_ylabel('Probability')

    ax.grid(linestyle='--', axis='x')
    ax.set_title(title)
    # plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
    plt.tight_layout()

def plot_graph_distr(g):
    fig, ax = plt.subplots()

    for v, distr in g.iter_vertices([g.vp.prior_distr]):
        ax.plot(bias(), distr, label=str(v))

    ax.set_xlabel(r'Bias, $\theta$')
    ax.set_ylabel('Probability')

    ax.grid(linestyle='--', axis='x')
    # plt.tick_params(axis='y', which='both', left=False, right=False, top=False, bottom=False, labelleft=False)
    plt.tight_layout()


# %%
def graph_get_prior_distr(g):
    """
    Returns matrix: 
        row i corresponding to the prior distribution for person i, x_i(t, θ)
        col j corresponding to the prior distribution evaluated at θ = 0.05 * j
    """
    return g.vp.prior_distr.get_2d_array(np.arange(21)).T


# %%
def init_simulation(g, prior_mean=None, prior_sd=None): # mean_range=(0.0, 1.0), fwhm_range=(0.2, 0.8)
    n = g.num_vertices(ignore_filter=True)

    # randomly generate prior and prior sd if no args given
    if prior_mean is None or prior_sd is None:
        prior_mean, prior_sd = generate_priors(n)
        print(prior_mean, prior_sd)
    
    # equivalent to g.vp.prior_mean.get_array()[:]
    g.vp.prior_mean.a = prior_mean
    g.vp.prior_sd.a = prior_sd

    # transpose is O(1) !!!
    distr = prior_distr_vec(n, prior_mean, prior_sd).T
    # required for indirect array access of vector
    g.vp.prior_distr.set_2d_array(distr)
    
    # print(g.vp.prior_distr.get_2d_array(np.arange(21)).T)
    """
    #  iteration might be slow for enumeration
    for v in g.iter_vertices():
        g.vp.prior_distr.a = distr[v]
    """
    


# %%
def step_simulation(g, true_bias=0.5, learning_rate=0.25):
    """
    true_bias (θ_0 in the paper)
    learning_rate (μ / μ_i in the paper)
    TODO later: learning_rate can be a vector, length # nodes
        sets a learning rate for each person in the graph.
    """
    ## simulate an independent coin toss
    toss = toss_coin()

    ## update the opinions of each agent according to Bayes' Theorem (observe coin toss)
    prior_distr = graph_get_prior_distr(g)
    norm_const = normalising_constant(coin_toss_likelihood(), prior_distr)
    likelihood = coin_toss_likelihood()[toss]    # (eqn 2)
    # Fix some broadcasting issue https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    posterior_distr = likelihood * prior_distr / norm_const[:,None]  # (eqn 1)
    # Rows: node i's posterior distribution.  Cols: posterior distr evaluated at theta

    ## Mix the opinions of each node with respective neighbours (eq 3,4)
    avg_dist_in_belief = avg_dist_in_belief(g, posterior_distr)

    next_dist = posterior_distr + avg_dist_in_belief * learning_rate

    # TODO: use np.amax to find max
    # TODO: calculate proportionality constant by eqn 3


# %%
init_simulation(g, np.array((0.25, 0.75)), np.array([fwhm_to_sd(0.4)] * 2))


# %%
def friendliness_mat(g):
    # get the friendliness matrix 
    # (graph_tool internally uses an adjacency list representation, 
    #  need to convert the weight list to a weight matrix; 
    #  rows / cols being neighbouring nodes)
    # laplacian doc: https://graph-tool.skewed.de/static/doc/spectral.html#graph_tool.spectral.laplacian
    friendliness = -gt.laplacian(g, weight=g.ep.friendliness).toarray()
    np.fill_diagonal(friendliness, 0) # diagonals of laplacian are some 'gamma'
    return friendliness


# %%
def avg_dist_in_belief(g, posterior):
    n = g.num_vertices(ignore_filter=True)

    friendliness = friendliness_mat(g)

    xi = np.broadcast_to(posterior_distr, (n, n, 21)).transpose((1, 0, 2))
    xj = np.broadcast_to(posterior_distr, (n, n, 21))
    # calculate the difference in belief for node i to node j
    diff_in_belief = ((xj - xi).T * friendliness).T
    summation = np.sum(diff_in_belief, axis=1)

    # sums over the columns
    divisor = np.reciprocal(np.sum(np.abs(friendliness), axis=1))

    # the .T stuff fixes some broadcasting issue  (same above too!)
    avg_dist_in_belief = (summation.T * divisor).T
    return avg_dist_in_belief


# %%
# OLD VERSION!
adj = gt.adjacency(g).todense()
xj = np.matmul(adj, posterior_distr)
xi = posterior_distr
diff_in_belief = xj - xi
diff_in_belief = np.broadcast_to(interactions, (n, n, 21))

friendliness = -gt.laplacian(g, weight=g.ep.friendliness).toarray()
np.fill_diagonal(friendliness, 0) # diagonals of laplacian are some 'gamma'

# sums over the columns
divisor = np.reciprocal(np.sum(np.abs(friendliness), axis=1))

# don't know how else to do this one, it would be nice if we didn't need
# to do this (particularly the [0] at the end!)
avg_dist_in_belief = np.tensordot(friendliness, interactions, axes=1)[0] * divisor


# %%
# TEST VERSION!
i = 1
j = 0
del_x1 = -(posterior_distr[j] - posterior_distr[i])
del_x0 = -(posterior_distr[i] - posterior_distr[j])
res = np.array([del_x0, del_x1])
res


# %%
plot_distr(posterior_distr, title="Posterior distr")


# %%
posterior_distr[1]


# %%
posterior_distr.shape


# %%
np.broadcast_to(posterior_distr, (2, 2, 21)) - posterior_distr


# %%
posterior_distr.shape


# %%
posterior_distr


# %%
[posterior_distr - posterior_distr[0], posterior_distr - posterior_distr[1]]


# %%



# %%
plot_distr(g)


# %%
debug_state(g)


# %%



