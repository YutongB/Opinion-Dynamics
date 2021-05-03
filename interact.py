# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'


# %%
from IPython import get_ipython
import cProfile
from sim import *
from datetime import datetime
%load_ext autoreload

# %%
import matplotlib as mpl
mpl.use('cairo')
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
draw_graph(complete_graph_of_random(10))

# %%
g = pair_of_allies()
init_simulation(g, prior_mean=np.array((0.25, 0.75)),
                prior_sd=np.array([fwhm_to_sd(0.4)] * 2))

draw_graph(g)

# %%
g = complete_graph_of_enemies(2)

with cProfile.Profile() as pr:
    seed(42)
    res = run_simulation(g,
                         max_steps=2000,
                         prior_mean=np.array((0.25, 0.75)),
                         prior_sd=np.array([fwhm_to_sd(0.4)] * 2),
                         tosses_per_iteration=10
                         )
    steps, asymptotic, coins, mean_std, distr, initial_distr = res
mean_std = np.array(mean_std)
steps, asymptotic

# 0.530 seconds (677 iterations) without graphtool storing priors
# 3.254 seconds (677 iterations) with graphtool

# %%
%%time
g = complete_graph_of_random(3)

init_simulation(g)
res = run_simulation(g, max_steps=1000)

# %%
g.vp.prior_mean.a


# %%
#pr.print_stats()


# %%
plt.plot(np.cumsum(coins))

# %%
# mean
plt.plot(mean_std[:, :, 0], alpha=0.5)

# %%
# stddev
plt.plot(mean_std[:, :, 1], alpha=0.5)

# %%
# initial distribution
plot_distr(initial_distr)

# %%
# final distribution
plot_distr(distr)


def timestamp():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

# %%
last_results_fname = "output/res-{}.json".format(timestamp())
dump_results(res, last_results_fname)

res_read = read_results(last_results_fname)

# works :)


# %%
last_graph_fname = "output/graph-{}.gt".format(timestamp())
g.save(last_graph_fname)

g = gt.load_graph(last_graph_fname)

draw_graph(g)

# %%
%%time
res = do_ensemble(runs=10, gen_graph=lambda: complete_graph_of_enemies(10), sim_params={"max_steps": 1000})


# %%
# %%time
# res = do_ensemble_parallel(runs=10, gen_graph=lambda: complete_graph_of_enemies(10), sim_params={"max_steps": 1000})



# %%
with cProfile.Profile() as pr:
    res = do_ensemble(runs=20, gen_graph=lambda: complete_graph_of_random(15), sim_params={"max_steps": 10000})

# %%
pr.print_stats()

# %%
%%time
res = do_ensemble(runs=20, gen_graph=lambda: complete_graph_of_friends(10), sim_params={"max_steps": 1000})

# %%
