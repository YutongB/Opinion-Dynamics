from cProfile import label
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import graph_tool as gt 
import plotly.express as px
import gif
from tqdm import tqdm

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

    def plot_coins_prob(self, label = None):
        sim = self.results
        sum = np.cumsum(sim.coins)
        prob =[]
        for index, value in enumerate(sum):
            prob.append(value/(index+1))
        prob.pop(0)
        # print(sum)
        plt.plot(prob, label = label)
        plt.xlabel("Step")
        plt.ylabel("Probability of heads")
    """
    Ploting the number of steps the system is in asymptotic learning consecutively
    Code will terminate after reaching 100 steps if "break_on_asymptotic_learning": True,
    """
    def plot_asymptotic_learning_steps(self):
        sim = self.results
        plt.plot(sim.asymptotic, label = "Asymptotic learning steps")
        plt.xlabel("Step")
        plt.ylabel("Number of consecutive asymptotic learning steps")

    def plot_asymptotic_learning_steps_hist(self,include_zero = True,bins = 50,log = False):
        sim = self.results
        if include_zero:
            plt.hist(sim.asymptotic, bins = bins, log=log)
        else: 
            new_step = []
            for step in sim.asymptotic:
                if step != 0:
                    new_step.append(step)
            plt.hist(new_step, bins = bins,log=log)
        plt.xlabel("Number of consecutive asymptotic learning steps")
        plt.ylabel("Frequency")

    def dwell_time(self): 
        sim = self.results
        # dwelltime = []
        steps = np.array(sim.asymptotic)
        dwell_index = np.where(np.diff(steps) < 0)[0]

        for i in dwell_index: # make sure that code above worked correctly
            assert (steps[i] > 0 and steps[i+1] == 0)

        dwell_time = [steps[i] for i in dwell_index]
        return dwell_time, dwell_index
    
    def plot_dwell_time(self, bins = 'auto', log = False):
        plt.hist(self.dwell_time(), label = "Dwell time", bins = bins, log = log)
        plt.xlabel("Dwell time")
        plt.ylabel("Frequency")

    def dt_at_true(self):
        sim = self.results
        true_theta = self.sim_params["true_theta"]
        

    def nic_dwell_time(self, stable_iters = None, stable = True):
        sim = self.results
        if stable_iters is None:
            stable_iters = self.sim_params["asymptotic_learning_max_iters"]
        steps = np.array(sim.asymptotic)
        indices_at_stable = np.nonzero(steps >= stable_iters)[0]
        start_time_list = [indices_at_stable[0]]
        end_time_list = []
        for i in range(len(indices_at_stable)-1):
            if indices_at_stable[i+1] - indices_at_stable[i] != 1:
                end_time_list.append(indices_at_stable[i])
                start_time_list.append(indices_at_stable[i+1])

        if stable:
            time_in_stable = [end_time_list[i]- start_time_list[i] for i in range(len(end_time_list))]
            return time_in_stable, start_time_list
            # return [(start_time_list[i], time_in_stable[i]) for i in range(len(time_in_stable))]
        else:
            time_in_turbulence = [ start_time_list[i+1] - end_time_list[i] for i in range(len(start_time_list)-1)]
            return time_in_turbulence, end_time_list
            # return [(end_time_list[i], time_in_turbulence[i]) for i in range(len(time_in_turbulence))]
            

    def plot_nic_dwell_time(self, stable_iters = None, stable = True, BINS = 'auto', log = False):
        dwell_time, _ = self.dwell_time(stable_iters,stable)
        if log:
            plt.hist(dwell_time, bins = BINS, log = True)
        plt.hist(dwell_time, bins = BINS)
        plt.xlabel("Step")
        plt.ylabel("Dwell time")
        if stable:
            plt.title("Dwell time in stable state")
        else:
            plt.title("Dwell time in turbulence")   



    '''
    If "break_on_asymptotic_learning": False, 
    check the belief of non-partisans when the asymptotic learning step is reached
    If "break_on_asymptotic_learning": True,
    It will return final belief of non-partisans
    '''
    def belief_at_asymptotic(self, stable_iters = None):
        sim = self.results
        steps = np.array(sim.asymptotic)
        if stable_iters is None:
            ASYM_MAX_ITERS = self.sim_params["asymptotic_learning_max_iters"]
        else: 
            ASYM_MAX_ITERS = stable_iters
        indices = np.nonzero(steps == ASYM_MAX_ITERS)[0]
        belief_at_asy = [sim.mean_list[i][-1] for i in indices]

        return belief_at_asy
    
    '''
    Check the belief at asymptotic learning, 
    retune the fraction of asymptotic learning steps with belief 
    that are close to the true bias/or given belief
    '''
    def check_belief_at_asymptotic(self, iter = None, belief = None):
        belief_arr = np.array(self.belief_at_asymptotic(iter))
        if belief is None:
            belief = self.sim_params["true_bias"]
        coin_arr = np.full((1,len(belief_arr)),belief)
        return np.sum(np.isclose(belief_arr, coin_arr,atol=0.05))/len(belief_arr)

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
    def plot_all_mean(self):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(sim.mean_list[:, 0], color="black", linestyle='dashed', linewidth=4)
        
        plt.plot(sim.mean_list[:, 1:], alpha=0.5)
        plt.grid(linestyle='dotted')

        # plt.title(f"Mean/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("$ \\langle \\theta\\rangle $")
        n = len(self.results.initial_distr)
        # plt.legend(range(n))

    def plot_non_partisan_mean(self,  LINEWIDTH= 0.5, ALPHA = 0.5, LABLE = None):
        sim = self.results
        # alpha is transparency of graph lines
        if LABLE == None: 
            plt.plot(sim.mean_list[:,-1], linewidth=LINEWIDTH, alpha=ALPHA)
        else: 
            plt.plot(sim.mean_list[:,-1], linewidth=LINEWIDTH, alpha=ALPHA, lable = LABLE)
            
        plt.xlabel("Timesteps")
        # plt.title("Mean belief of one non-partisan agent")
        plt.ylabel("$\\langle \\theta \\rangle$")


    def plot_std(self):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(sim.std_list, alpha=0.5)
        plt.title(f"StdDev/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Standard Deviation")
        n = len(self.results.initial_distr)
        plt.legend(range(n))


    def plot_distr(self, step, title=None, simid=None):
        sim = self.results
        # alpha is transparency of graph lines
        if simid is None:
            plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T, linewidth=1)
        else:
            plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step][simid].T, linewidth=1)
        # plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T,marker='x')

        if title is None:
            title = f"Distribution step {step}"

        # plt.title(f"{title}, sim {self.idx}")
        plt.xlabel("$\\theta$")
        plt.ylabel("Probability")
        n = len(self.results.initial_distr)
        # plt.legend(range(n))

    """
    Plotly slider plot
    """

    def plotly_distr(self, steps=None, opacity=1):
        theta = np.linspace(0, 1, BIAS_SAMPLES)
        if steps is None:
            steps = range(self.results.distrs.shape[0])
        steps = list(steps)


        df = pd.DataFrame([(v, theta[i], node, steps[step]) 
                            for step, ds in enumerate(self.results.distrs[steps])
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
Fitting and plotting 
'''
'''
Powerlaw fit
'''
    
def get_xy_from_hist(data, exclude = 0):   
    counts = np.unique(data, return_counts=True)
    x = counts[0]
    y = counts[1]
    exclude = int(len(x) * (1 - exclude / 100))
    x = counts[0][:exclude]
    y = counts[1][:exclude]
    return x, y



def fit_with_exclude(data, log=True, exclude=40):
    """find line best fit with given exclude parameters"""
    x, y = get_xy_from_hist(data, exclude=exclude)
    exclude = int(len(x) * (1 - exclude / 100))
    scaledx, scaledy = x, y
    if log:
        scaledx = x
        scaledy = np.log10(y)
    c, m = np.polyfit(scaledx[:exclude], scaledy[:exclude], 1)
    fittedy = m * x + c
    equation = "y = {:.4f} x + {:.4f}".format(m, c)
    if log:
        fittedy = 10 ** fittedy
        equation = "log y = {:.4f} x + {:.4f}".format(m, c)
    return fittedy, equation

def plot_scatter(data, xlabel= None, ylabel = None, title =  None, fit=True, log=True, exclude=40):
    """scatter plot with line best fit"""
    x, y = get_xy_from_hist(data)
    fig, ax = plt.subplots()
    if log:
        # ax.set_xscale('log')
        ax.set_yscale('log')
    # ax.hist(data)
    # ax.scatter(x, y)
    if fit:
        fittedy, equation = fit_with_exclude(data, exclude= exclude)
        exclude = int(len(x) * (1 - exclude / 100))
        ax.plot(x[:exclude], fittedy[:exclude], '-')
        print(title, equation)
        ax.text(0.1, 0.1, equation, transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()



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
        theta0_close_agents = np.isclose(sim.mean_list[-1], theta0, atol=1e-2)
        thetap_close_agents = np.isclose(sim.mean_list[-1], thetap, atol=1e-2)
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





# def powerlaw_fit(x, y, xmin=None, xmax=None, **kwargs):
#     popt, pcov = curve_fit(powerlaw_func, x[:20], y[:20], p0=[5000, 0.5, 0], bounds=([1,.05, -2000], [10000, 5, 2000]))