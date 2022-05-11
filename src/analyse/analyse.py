from typing import List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import graph_tool as gt 
import plotly.express as px
import gif

from src.analyse.results import SimResults, read_graph, read_results
from src.simulation.sim import BIAS_SAMPLES

def indexof_belief(belief):
    try:
        return [round(x, 2) for x in np.linspace(0, 1, BIAS_SAMPLES)].index(belief)
    except ValueError:
        raise ValueError(f"Invalid belief θ={belief}")
    
class AnalyseSimulation:
    def __init__(self, results: SimResults, args, sim_params, idx: int):
        self.results: SimResults = results
        self.args = args
        self.sim_params = sim_params
        self.idx = idx
        self.disruption = args['disruption']
        self.n = args['size']
        self.asymptotic = self.results.asymptotic[-1] == self.sim_params["asymptotic_learning_max_iters"]

    '''
    Functions for drawing graphs
    '''
    def graph(self):
        return read_graph(self.results.adjacency, self.results.friendliness)

    def show_graph(self):
        gt.draw.graphviz_draw(self.graph)

    def show_params(self):
        print("sim params ------")
        for k, v in self.sim_params.items():
            print(f"{k} - {v}")
    '''
    Ploting the coin toss
    Number of heads vs step
    '''
    def plot_coins(self):
        sim = self.results
        plt.plot(np.cumsum(sim.coins))
        plt.xlabel("Step")
        plt.ylabel("Number of heads")
    """
    Ploting the number of steps the system is in asymptotic learning consecutively
    Code will terminate after reaching 100 steps
    """
    def plot_asymptotic_learning_steps(self):
        sim = self.results
        plt.plot(sim.asymptotic)
        plt.xlabel("Step")
        plt.ylabel("Number of consecutive asymptotic learning steps")

    """
    Ploting confidence in beliefs in all different ways 
    """

    def plot_confidence_in_belief(self, belief: float, opacity=.8):
        self.plot_confidence_in_belief_sum([belief], opacity=opacity)

    def calc_confidence_in_belief(self, beliefs: List[float]):
        # belief (theta)
        sim = self.results
        beliefs_idx = [indexof_belief(belief) for belief in beliefs]        

        distr = sim.distrs.T[beliefs_idx].T # shape: (steps, n, len(beliefs))
        return distr

    def plot_confidence_in_belief_sum(self, beliefs: List[float], opacity=.8):
        plt.plot(np.sum(self.calc_confidence_in_belief(beliefs), axis=2), alpha=opacity)
        if len(beliefs) == 1:
            plt.title(f"Confidence in belief θ={beliefs[0]}, sim {self.idx}")
        else:
            plt.title(f"Confidence in sum of beliefs θ={beliefs}, sim {self.idx}")
        plt.xlabel("Step")
        plt.ylabel("Confidence")
        n = len(self.results.initial_distr)
        plt.legend(range(n), bbox_to_anchor=(1.04,1), loc="upper left")

    def plot_confidence_in_belief_overall(self, beliefs: List[float], opacity=.8):
        n = len(self.results.initial_distr)
        plt.plot(np.sum(np.sum(self.calc_confidence_in_belief(beliefs), axis=2), axis=1) / n, alpha=opacity)
        if len(beliefs) == 1:
            plt.title(f"Confidence in belief θ={beliefs[0]}, sim {self.idx}")
        else:
            plt.title(f"Confidence in sum of beliefs θ={beliefs}, sim {self.idx}")
        plt.xlabel("Step")
        plt.ylabel("Confidence")

    def plot_confidence_in_beliefs_and_sum(self, beliefs: List[float], opacity=.8):
        for belief in beliefs:
            distr = self.calc_confidence_in_belief([belief])[:, -1, :]
            plt.plot(distr, alpha=opacity, label = f"$\\theta = {belief}$")
        n = len(self.results.initial_distr)
        plt.plot(np.sum(np.sum(self.calc_confidence_in_belief(beliefs), axis=2), axis=1) / n, alpha=opacity, label = "Sum of all beliefs" )

        plt.title(f"Confidence in beliefs θ={beliefs} and sum of beliefs, sim {self.idx}")
        plt.xlabel("Step")
        plt.ylabel("Confidence")
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


    '''
    Ploting system statistics 
    '''
    def plot_mean(self):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(sim.mean_list, alpha=0.5)
        plt.title(f"Mean/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Mean")
        n = len(self.results.initial_distr)
        plt.legend(range(n))

    def plot_std(self):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(sim.std_list, alpha=0.5)
        plt.title(f"StdDev/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Standard Deviation")
        n = len(self.results.initial_distr)
        plt.legend(range(n))


    def plot_distr(self, step, title=None):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T)

        if title is None:
            title = f"Distribution step {step}"

        plt.title(f"{title}, sim {self.idx}")
        plt.xlabel("$\\theta$")
        plt.ylabel("Probability")
        n = len(self.results.initial_distr)
        plt.legend(range(n))

    """
    Plotly slider plot
    """

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

    def plotly_distr_frame(self, step, opacity=1):
        theta = np.linspace(0, 1, BIAS_SAMPLES)

        df = pd.DataFrame([(v, theta[i], node, step) 
                            for node, row in enumerate(self.results.distrs[step]) 
                            for i, v in enumerate(row)], 
                    columns=["y", 'theta', 'node', 'step'])

        fig = px.line(df, x="theta", y="y", color="node", title=f"Belief distribution at step {step}")
        fig.update_traces(opacity=opacity)
        return fig

    def plotly_distr_gif(self):
        num_steps = self.results.distrs.shape[0]
        frames = [gif.frame(self.plotly_distr_frame)(step) for step in range(num_steps)]
        fname = 'plots/' + get_clean_last_results_fname() + '.gif'

        gif.save(frames, fname, duration=1, unit = 's', between="startend")

    def plot_initial_distr(self):
        self.plot_distr(0, "Initial distribution")

    def plot_final_distr(self):
        self.plot_distr(-1, "Final distribution")

    def indices_of_change(self, beliefs, threshold=500):
        # we are only interested in the unfixed (last) node
        confidence = self.calc_confidence_in_belief(beliefs)[:,-1,:]
        diff = confidence[:,0] - confidence[:,1]

        indices_of_change = np.where(diff[:-1] * diff[1:] < 0)[0]
        indices_of_change = indices_of_change[indices_of_change > threshold]
        dist_between_changes = indices_of_change[1:] - indices_of_change[:-1]
        # num_changes = len(indices_of_change)
        return indices_of_change, dist_between_changes

    def plot_dist_between_changes(self, beliefs, threshold=500):
        _, dist_between_changes = self.indices_of_change(beliefs, threshold=threshold)
        plt.hist(dist_between_changes, bins=60)
        plt.xlabel("Number of steps between change")
        plt.ylabel("Frequency")





'''
Write result into file
Putting last result into folder for easier access
'''

def analyse_results(filename):
    out = read_results(filename)
    sim_params, args = out['sim_params'], out['args']
    return [AnalyseSimulation(res, args, sim_params, i) for i, res in enumerate(out['results'])]

def analyse_last_results():
    return analyse_results(get_last_results_fname())

def get_last_results_fname():
    with open("output/last_results", 'r') as f:
        return f.read()

def get_clean_last_results_fname():
    return get_last_results_fname().split('/')[-1].split('.')[0]


def frac_simulation_asymptotic(results):
    ensemble_t_A = [sim.results.steps if sim.asymptotic else -1 for sim in results]
    num_asymptotic = len(ensemble_t_A) - ensemble_t_A.count(-1)
    pct_asymptotic = num_asymptotic / len(ensemble_t_A)
    return " ".join(map(str,ensemble_t_A)), pct_asymptotic, "{:.2f}% ({}/{}) asymptotic".format(pct_asymptotic*100, num_asymptotic, len(ensemble_t_A))


def num_asymptotic_agents_theta(ensemble: List[SimResults], theta0: float, thetap: float):
    res = []
    for sim in ensemble:
        asymp_agents = np.array(sim.agent_is_asymptotic)
        num_asymp_agents = np.sum(asymp_agents)
        theta0_close_agents = np.isclose(sim.mean_list, theta0, atol=1e-2)
        thetap_close_agents = np.isclose(sim.mean_list, thetap, atol=1e-2)
        not_close_agents = ~(theta0_close_agents | thetap_close_agents)

        theta0_close_asymp_agents = theta0_close_agents & asymp_agents
        thetap_close_asymp_agents = thetap_close_agents & asymp_agents
        not_close_asymp_agents = not_close_agents & asymp_agents
        
        res.append((np.sum(theta0_close_asymp_agents) / num_asymp_agents,
                   np.sum(thetap_close_asymp_agents) / num_asymp_agents,
                   np.sum(not_close_asymp_agents) / num_asymp_agents))

    return list(np.array(res).mean(axis=0))  # take the mean across the ensemble


def num_asymptotic_agents(ensemble: List[SimResults]):
    num_asymptotic_agents = [np.sum(sim.agent_is_asymptotic) for sim in ensemble]
    return np.mean(num_asymptotic_agents)


def frac_asymptotic_system(ensemble: List[SimResults], asymp_max_iters: int):
    ensemble_t_A = [sim.steps if sim.asymptotic[-1] == asymp_max_iters else -1 for sim in ensemble]
    num_asymptotic = len(ensemble_t_A) - ensemble_t_A.count(-1)
    frac_asymptotic = num_asymptotic / len(ensemble_t_A)
    return frac_asymptotic

