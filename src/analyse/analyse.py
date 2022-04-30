from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import plotly.express as px

from src.analyse.results import read_graph, read_results
from src.simulation.sim import BIAS_SAMPLES

class AnalyseSimulation:
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx
        self.n = len(data.initial_distr.T)

    def graph(self):
        return read_graph(self.data.adjacency, self.data.friendliness)

    def plot_coins(self):
        sim = self.data
        plt.plot(np.cumsum(sim.coins))
        plt.legend(range(self.n))

    def plot_mean(self):
        sim = self.data
        # alpha is transparency of graph lines
        plt.plot(sim.mean_list, alpha=0.5)
        plt.title(f"Mean/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Mean")
        plt.legend(range(self.n))

    def plot_std(self):
        sim = self.data
        # alpha is transparency of graph lines
        plt.plot(sim.std_list, alpha=0.5)
        plt.title(f"StdDev/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Standard Deviation")
        plt.legend(range(self.n))


    def plot_distr(self, step, title=None):
        sim = self.data
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
                            for step, ds in enumerate(self.data.distrs) 
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
    return [AnalyseSimulation(res, i) for i, res in enumerate(read_results(filename))]

def analyse_last_results():
    with open("output/last_results", 'r') as f:
        last_results_fname = f.read()

        return analyse_results(last_results_fname)
