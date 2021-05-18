# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'


# %%
from IPython import get_ipython
import cProfile
from sim import *

import matplotlib as mpl
mpl.use('cairo')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

class AnalyseResults:
    DEF_IDX = -1
    def __init__(self, filename):
        self.filename = filename
        """ a list of results
        SimResults = namedtuple("SimulationResults",
                                ("steps", "asymptotic", "coins", "mean_list", "std_list", 
                                "final_distr", "initial_distr", "friendliness", "adjacency"))
        """
        self.results = read_results(filename)

    def sim(self, sim_idx=DEF_IDX):
        return self.results[sim_idx]

    def graph(self, sim_idx=DEF_IDX):
        sim = self.sim(sim_idx)
        return read_graph(sim.adjacency, sim.friendliness)
    
    def draw_graph(self, sim_idx=DEF_IDX):
        draw_graph(self.graph(sim_idx))

    def friendliness(self, sim_idx=DEF_IDX):
        return self.sim(sim_idx).friendliness

    def adjacency(self, sim_idx=DEF_IDX):
        return self.sim(sim_idx).adjacency
    
    def plot_coins(self, sim_idx=DEF_IDX):
        sim = self.sim(sim_idx)
        n = len(sim.initial_distr.T)
        plt.plot(np.cumsum(sim.coins))
        plt.legend(range(n))

    def plot_mean(self, sim_idx=DEF_IDX):
        sim = self.sim(sim_idx)
        n = len(sim.initial_distr.T)
        # alpha is transparency of graph lines
        plt.plot(sim.mean_list, alpha=0.5)
        plt.xlabel("Iteration")
        plt.ylabel("Mean")
        plt.legend(range(n))

    def plot_std(self, sim_idx=DEF_IDX):
        sim = self.sim(sim_idx)
        n = len(sim.initial_distr.T)
        # alpha is transparency of graph lines
        plt.plot(sim.std_list, alpha=0.5)
        plt.xlabel("Iteration")
        plt.ylabel("Standard Deviation")
        plt.legend(range(n))

    def plot_initial_distr(self, sim_idx=DEF_IDX):
        sim = self.sim(sim_idx)
        n = len(sim.initial_distr.T)
        # alpha is transparency of graph lines
        plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.initial_distr.T)
        plt.xlabel("$\\theta$")
        plt.ylabel("Probability")
        plt.legend(range(n))

    def plot_final_distr(self, sim_idx=DEF_IDX):
        sim = self.sim(sim_idx)
        n = len(sim.initial_distr.T)
        # alpha is transparency of graph lines
        plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.final_distr.T)
        plt.xlabel("$\\theta$")
        plt.ylabel("Probability")
        plt.legend(range(n))

    def get_col(self, colname):
        return list(map(lambda x: x._asdict()[colname], self.results))

    def steps_asymp(self):
        d = {
            "steps": self.get_col("steps"),
            "asymptotic": self.get_col("asymptotic"),
        }
        return pd.DataFrame(d)
# %%
res = AnalyseResults("output/res-2021_05_18-23_42_23.json")

# %%
res.plot_initial_distr()
# %%
res.plot_final_distr()

# %%
res.plot_mean()

# %%
res.plot_std()


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
