
""" Rich progress bars """

import signal
from threading import Event
from typing import List
import numpy as np

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from src.analyse.results import SimResults

from src.simulation.initsim import init_simulation
from src.simulation.sim import adjacency_mat, friendliness_mat, mean_distr, std_distr, step_simulation

done_event = Event()
def handle_sigint(signum, frame):
    done_event.set()

signal.signal(signal.SIGINT, handle_sigint)

def make_progress():
    return Progress(
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=15),
        "[progress.percentage]{task.completed}",
        "/",
        "{task.total}",
        "•",
        TimeElapsedColumn(),
        "/",
        TimeRemainingColumn(),
        refresh_per_second=5,
        transient=True,
    )

""" End rich progress bars """


def run_simulation(g, max_steps=1e4, asymptotic_learning_max_iters=99,
                   prior=None,
                   true_bias=0.5, learning_rate=0.25,
                   tosses_per_iteration=1, task_id=None, progress=None,
                   log=None, DWeps=1, coinslist=None,
                   disruption = 0,
                   break_on_asymptotic_learning = True,
                   hide_progress_after_complete = True,
                   title="Simulation") -> SimResults:
    """
    max_steps (T in the paper) - maximum number of steps to run the simulation
    """
    initial_distr = init_simulation(g, prior=prior)
    n = prior['n']
    distrs = []
    coins_list = []
    mean_list = []
    std_list = []
    asymptotic = []
    iters_asymptotic_learning = 0
    iters_asymptotic_learning_agents = [0] * n
    prior_distr = initial_distr.copy()
    distrs.append(prior_distr)

    max_steps = int(max_steps)

    # For now, static friendliness; TODO: update friendliness dynamically. (Hi Christine!)
    friendliness = friendliness_mat(g)
    init_friendliness = friendliness

    if progress is None:
        progress = make_progress()

    if progress:
        if task_id is None:
            task_id = progress.add_task(title, total=max_steps)
        else:
            progress.start_task(task_id)
    
        progress.update(task_id, total=max_steps)

    steps = 0
    # steps 2-3 of Probabilistic automaton, until t = T
    for i in range(max_steps):

        coins = None if coinslist is None else coinslist[i]

        coins, posterior = step_simulation(
            g, prior_distr=prior_distr, true_bias=true_bias, learning_rate=learning_rate,
            friendliness=friendliness,
            num_coins=tosses_per_iteration,
            DWeps=DWeps, coins=coins, 
            disruption=disruption)

        steps += 1
        mean = mean_distr(posterior)
        mean_list.append(mean)
        if log is not None:
            std_list.append(std_distr(posterior, mean))
            coins_list.append(coins)
            distrs.append(posterior)
        
        # system reaches asymptotic learning when all agents reach asymptotic learning   
        largest_change = np.max(np.abs(posterior - prior_distr), axis=1)
        largest_peak = 0.01 * np.max(prior_distr, axis=1)
        agent_is_asymptotic = largest_change < largest_peak  # one element for each agent
        for i in range(len(agent_is_asymptotic)):  # loop over each agent
            if agent_is_asymptotic[i]:
                iters_asymptotic_learning_agents[i] += 1
            else:
                iters_asymptotic_learning_agents[i] = 0
        
        # for the whole system...
        all_is_asymptotic = np.all(agent_is_asymptotic)
        if all_is_asymptotic:
            iters_asymptotic_learning += 1
        else:
            iters_asymptotic_learning = 0
        asymptotic.append(iters_asymptotic_learning)

        if progress:
            progress.update(task_id, advance=1)

        if iters_asymptotic_learning == asymptotic_learning_max_iters:
            if progress:
                progress.update(task_id, total=i+1, completed=i+1)
            if break_on_asymptotic_learning:
                break

        prior_distr = posterior.copy()

        if done_event.is_set():
            print("!! Caught interrupt.")
            return

    adjacency = adjacency_mat(g)
    if progress:
        progress.update(task_id, visible=not hide_progress_after_complete)

    # agent_is_asymptotic = np.array(iters_asymptotic_learning_agents) >= asymptotic_learning_max_iters
    agent_is_asymptotic = np.array(iters_asymptotic_learning_agents)

    if log is None:
        return SimResults(steps=steps,
                          asymptotic=asymptotic,
                          agent_is_asymptotic=agent_is_asymptotic,
                          coins=None,
                          mean_list=mean_list[-1],
                          std_list=None,
                          final_distr=None,
                          initial_distr=None,
                          adjacency=None,
                          friendliness=None,
                          distrs=None)
    else:
        return SimResults(steps=steps,
                        asymptotic=asymptotic,
                        agent_is_asymptotic=agent_is_asymptotic,
                        coins=coins_list,
                        mean_list=mean_list,
                        std_list=std_list,
                        final_distr=prior_distr,
                        initial_distr=initial_distr,
                        adjacency=adjacency,
                        friendliness=friendliness,
                        distrs=distrs)


def run_ensemble(runs: int, gen_graph, sim_params=None, title="Ensemble", simulation_progress=True) -> list[SimResults]:
    """
    sim_params: dictionary of parameters to pass to run_simulation
    eg: sim_params = { "max_steps": 100 }
    """
    if runs < 1:
        raise Exception("invalid number of runs")

    if sim_params is None:
        sim_params = {}

    try:
        coinslists = sim_params["coinslists"]
        del sim_params["coinslists"]
        priors = sim_params["priors"]
        del sim_params["priors"]
    except:
        coinslists = None
        priors = None


    with make_progress() as progress:
        results = []
        ensemble = progress.add_task(title, total=runs)

        for r in range(runs):
            if simulation_progress:
                task_id = progress.add_task("Sim #{}".format(r+1))

            if coinslists is not None:
                sim_params["coinslist"] = coinslists[r]
                sim_params["prior"] = priors[r]

            sim = run_simulation(gen_graph(), **sim_params,
                                 progress=progress if simulation_progress else False,
                                 task_id=task_id if simulation_progress else None)
            results.append(sim)
            progress.update(ensemble, advance=1)

    return results



