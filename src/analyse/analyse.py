from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import plotly.express as px

from src.analyse.results import SimResults, read_graph, read_results
from src.simulation.sim import BIAS_SAMPLES

class AnalyseSimulation:
    def __init__(self, results: SimResults, args, sim_params, idx: int):
        self.results = results
        self.args = args
        self.sim_params = sim_params
        self.idx = idx
        self.n = len(results.initial_distr)

    def graph(self):
        return read_graph(self.results.adjacency, self.results.friendliness)

    def show_params(self):
        print("sim params ------")
        for k, v in self.sim_params.items():
            print(f"{k} - {v}")

    def plot_coins(self):
        sim = self.results
        plt.plot(np.cumsum(sim.coins))
        plt.xlabel("Step")
        plt.ylabel("Number of heads")

    def plot_confidence_in_belief(self, belief: float, opacity=.8):
        # belief (theta)
        sim = self.results
        try:
            belief_idx = list(np.linspace(0, 1, BIAS_SAMPLES)).index(belief)
        except ValueError:
            raise ValueError(f"Invalid belief θ={belief}")

        plt.plot(sim.distrs.T[belief_idx].T, alpha=opacity)
        plt.title(f"Confidence in belief θ={belief}, sim {self.idx}")
        plt.xlabel("Step")
        plt.ylabel("Confidence")
        plt.legend(range(self.n))


    def plot_mean(self):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(sim.mean_list, alpha=0.5)
        plt.title(f"Mean/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Mean")
        plt.legend(range(self.n))

    def plot_std(self):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(sim.std_list, alpha=0.5)
        plt.title(f"StdDev/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Standard Deviation")
        plt.legend(range(self.n))


    def plot_distr(self, step, title=None):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T)

        if title is None:
            title = f"Distribution step {step}"

        plt.title(f"{title}, sim {self.idx}")
        plt.xlabel("$\\theta$")
        plt.ylabel("Probability")
        plt.legend(range(self.n))

    def plotly_distr(self, opacity=1):
        theta = np.linspace(0, 1, BIAS_SAMPLES)

        df = pd.DataFrame([(v, theta[i], node, step) 
                            for step, ds in enumerate(self.results.distrs) 
                            for node, row in enumerate(ds) 
                            for i, v in enumerate(row)], 
                    columns=["y", 'theta', 'node', 'step'])

        fig = px.line(df, x="theta", y="y", animation_frame="step", 
                    color="node", hover_name="node",
                )
        fig.update_traces(opacity=opacity)
        fig.show()

    def plot_initial_distr(self):
        self.plot_distr(0, "Initial distribution")

    def plot_final_distr(self):
        self.plot_distr(-1, "Final distribution")

def analyse_results(filename):
    out = read_results(filename)
    sim_params, args = out['sim_params'], out['args']
    return [AnalyseSimulation(res, args, sim_params, i) for i, res in enumerate(out['results'])]

def analyse_last_results():
    with open("output/last_results", 'r') as f:
        last_results_fname = f.read()

        return analyse_results(last_results_fname)
