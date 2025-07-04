from cProfile import label
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import graph_tool as gt 
import math

from .results import SimResults, read_graph, read_results, dump_json
from ..simulation.stat import bias
from ..simulation.graphs import show_graph
from ..utils import timestamp

def indexof_belief(belief, bias):
    try:
        return [round(x, 2) for x in bias].index(belief)
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

    def dump(self):
        fname = "output/res-{}.json".format(timestamp())
        out = dict(results=self.results, args=self.args, sim_params=self.sim_params)
        dump_json(out, fname)
        print("Wrote latest results to", fname)
        with open('output/last_results', 'w') as f:
            f.write(fname)

    '''
    Functions for drawing graphs
    '''
    def graph(self):
        return read_graph(self.results.adjacency, self.results.friendliness)

    def show_graph(self):
        show_graph(self.graph(), partisans=self.disruption)

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
        plt.plot(prob, label = label, linewidth = 0.7, alpha = 0.8)
        plt.xlabel("Timesteps")
        plt.ylabel("Estimated probability of heads")
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

    def dwell_time(self, return_index = False): 
        sim = self.results
        # dwelltime = []
        steps = np.array(sim.asymptotic)
        # last index at dwell
        dwell_index = np.where(np.diff(steps) < 0)[0]

        for i in dwell_index: # make sure that code above worked correctly
            assert (steps[i] > 0 and steps[i+1] == 0)

        dwell_time = [steps[i] for i in dwell_index]
        if return_index:
            return dwell_time, dwell_index
        return dwell_time
    

    def plot_dwell_time(self, bins = 100, log = True, label = None):
        plt.hist(self.dwell_time(), bins = bins, weights = np.ones_like(self.dwell_time())/len(self.dwell_time()), density = False, log = log, histtype="step", alpha = 1,label = label, stacked= True)
        plt.xlabel("$ t_{\\rm d}$")
        plt.ylabel("Normalized Number")


    def plot_dwell_time_wrt_belief(self, bins = 60, log = True, label = None):
        dwell_time, dwell_index = self.dwell_time(return_index = True)
        belief_at_dwell = [self.results.mean_list[i][-1] for i in dwell_index]
        plt.hist(belief_at_dwell, weights =  np.ones_like(belief_at_dwell)/len(belief_at_dwell),bins = bins, density=False,log = log, histtype="bar", label = label, stacked = True)
        plt.xlabel("$ \\langle \\theta \\rangle$")
        plt.ylabel("Normalized Number")


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
        beliefs_idx = [indexof_belief(belief, self.args.get("bias", bias())) for belief in beliefs]        

        distr = sim.distrs.T[beliefs_idx].T # shape: (steps, n, len(beliefs))
        return distr

    def plot_confidence_in_belief_sum(self, beliefs: List[float], opacity=.8):

        plt.plot(np.sum(self.calc_confidence_in_belief(beliefs)[1:], axis=2), alpha=opacity, linewidth=1)
        # if len(beliefs) == 1:
        #     plt.title(f"Confidence in belief θ={beliefs[0]}, sim {self.idx}")
        # else:
        #     plt.title(f"Confidence in sum of beliefs θ={beliefs}, sim {self.idx}")
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

    def plot_confidence_in_beliefs_and_sum(self, beliefs: List[float], opacity=.8, agent=-1):
        for belief in beliefs:
            distr = self.calc_confidence_in_belief([belief])[:, agent, :]
            plt.plot(distr, alpha=opacity, label = f"$\\theta = {belief}$",  linewidth = 0.9)
        n = len(self.results.initial_distr)
        plt.plot(np.sum(np.sum(self.calc_confidence_in_belief(beliefs), axis=2), axis=1) / n, alpha=opacity, label = "Sum" , linewidth = 1)

        # plt.title(f"Confidence in beliefs θ={beliefs} and sum of beliefs, sim {self.idx}")
        plt.xlabel("Timesteps")
        plt.ylabel("$x_i(t, \\theta)$")
        plt.legend()
        # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")


    '''
    Ploting system statistics 
    '''
    def plot_all_mean(self):
        sim = self.results
        n = len(self.results.initial_distr)
        for i in range(n):
            if i in self.disruption:
                plt.plot(sim.mean_list[:, i], color="black", linestyle='dashed', linewidth=4,label="Partisan")
            else:
                plt.plot(sim.mean_list[:, i], linewidth=0.5, label=f"Agent {i}")

        plt.grid(linestyle='dotted')

        # plt.title(f"Mean/Iter, sim {self.idx}")
        plt.xlabel("Timesteps")
        plt.ylabel("$ \\langle \\theta\\rangle $")
        plt.legend()

    def plot_agent_mean(self, agent_id, steps=10000-1,linewidth = 0.5, color=None, linestyle = '-', alpha = 1, label=None):
        sim = self.results
        plt.plot(sim.mean_list[:steps, agent_id], alpha=1, linewidth=linewidth, label=label, color=color, linestyle = linestyle)
        plt.grid(linestyle='dotted')
        plt.xlabel("Timesteps")
        plt.ylabel("$ \\langle \\theta\\rangle $")

    def plot_std(self):
        sim = self.results
        # alpha is transparency of graph lines
        plt.plot(sim.std_list, alpha=0.5)
        plt.title(f"StdDev/Iter, sim {self.idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Standard Deviation")
        n = len(self.results.initial_distr)
        # plt.legend(range(n))


    def plot_distr(self, step, title=None, color=None, BIAS_SAMPLES = 21, label = label):
        sim = self.results
        n = len(self.results.initial_distr)
        for i in range(n):
            if i in self.disruption:
                plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T[:, i], color="black", linestyle='dashed', linewidth=2, label="Partisan")
            else:
                plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T[:, i], linewidth=0.5, label=f"Agent {i}")

        # plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T, linewidth = 1, alpha=0.8)

        # plt.plot(np.linspace(0, 1, BIAS_SAMPLES), sim.distrs[step].T,marker='x')

        if title is None:
            title = f"Distribution step {step}"
        plt.grid(linestyle='dotted')
        # plt.title(f"{title}, sim {self.idx}")
        plt.xlabel("$\\theta$")
        plt.ylabel("Probability")
        plt.legend()

    def get_max_belief(self, agent_id, BIAS_SAMPLES = 21, step_from = 0):
        sim = self.results
        max_belief = []
        for step in range(step_from, len(sim.distrs)):
            index = np.argmax(sim.distrs[step][agent_id])
            max_belief.append(np.linspace(0, 1, BIAS_SAMPLES)[index])
        return max_belief

    """
    Plotly slider plot
    """

    def plotly_distr(self, steps=None, opacity=1, BIAS_SAMPLES = 21):
        import plotly.express as px

        theta = np.linspace(0, 1, BIAS_SAMPLES)
        if steps is None:
            steps = range(self.results.distrs.shape[0])
        steps = list(steps)

        df = pd.DataFrame([(v, theta[i], node, steps[step]) 
                            for step, ds in enumerate(self.results.distrs[steps])
                            for node, row in enumerate(ds) 
                            for i, v in enumerate(row)], 
                    columns=["probability", 'theta', 'agent', 'step'])

        fig = px.line(df, x="theta", y="probability", animation_frame="step", 
                    color="agent", hover_name="agent",
                )
        fig.update_traces(opacity=opacity)
        fig.show()

    def gif_distr(self, path, last_step = None, legend = True):
        """ 
        Create a gif of the belief distribution over time using plotly
        last step: if None, use all steps, otherwise use up to last_step (inclusive)
        all steps may be slow 
        """
        import gif
        if last_step is None:
            num_steps = self.results.distrs.shape[0]
        else:
            num_steps = last_step + 1

        opacity=1
        BIAS_SAMPLES = 21
        @gif.frame
        def plot(step):
            theta = np.linspace(0, 1, BIAS_SAMPLES)

            df = pd.DataFrame([(v, theta[i], node, step) 
                                for node, row in enumerate(self.results.distrs[step]) 
                                for i, v in enumerate(row)], 
                        columns=["y", 'theta', 'node', 'step'])

            plt.figure()
            for node, group in df.groupby('node'):
                if node in self.disruption:
                    plt.plot(group['theta'], group['y'], label='Partisan', alpha=opacity, color='black', linestyle='dashed', linewidth=2)
                else:
                    plt.plot(group['theta'], group['y'], label=f'Node {node}', alpha=opacity)

            plt.title(f"Belief distribution at step {step}")
            plt.xlabel('$\\theta$')
            plt.ylabel('Probability')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            if legend:
                plt.legend()
            plt.tight_layout()
    
        frames = [plot(step) for step in range(num_steps)]
        gif.save(frames, path, duration=1)

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
    Right and wrong t_A
    '''
    def agent_t_A(self):
        sim = self.results
        agent_asym_time = sim.steps - sim.agent_is_asymptotic
        return agent_asym_time 

                
    def get_right_and_wrong(self):
        t_right = []
        t_wrong = []
        sim = self.results
        t_A_list =  list(zip(self.agent_t_A(), sim.mean_list[-1]))
        # t_right, t_wrong = [],[]
        for (ta, mean) in t_A_list:
            if math.isclose(mean, 0.6, abs_tol=0.001):
                t_right.append(ta)
            else:
                t_wrong.append(ta)

        return t_right, t_wrong

    def plot_right_and_wrong(self, density = True, bins = 'auto'):
        bt_right, bt_wrong = self.get_right_and_wrong()
        plt.hist(bt_right, density=density, bins = bins, alpha = .8, color="g")
        plt.hist(bt_wrong, density=density, bins = bins, alpha = .8, color="r")

    
    def get_dist_and_step(self, threshold= 0.9, source = 0, theta = 0.6):
        sim = self.results
        sim_graph = read_graph(sim.adjacency, sim.friendliness)
        dist, _ = gt.search.dijkstra_search(sim_graph, sim_graph.edge_properties.friendliness, source=source)
        dist_per_node = list(map(int, dist.get_array()))
        pr_list = [np.sum(np.ndarray.flatten(self.calc_confidence_in_belief([theta])[:,i]) >= threshold) for i in range(len(self.results.initial_distr))]
        return sorted(list(zip(dist_per_node, pr_list)))

        



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


