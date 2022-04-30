
""" Rich progress bars """

import signal
from threading import Event
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
        # transient=True,
    )

""" End rich progress bars """


def run_simulation(g, max_steps=1e4, asymptotic_learning_max_iters=10,
                   prior=None,
                   true_bias=0.5, learning_rate=0.25,
                   tosses_per_iteration=10, task_id=None, progress=None,
                   log=None, DWeps=1, coinslist=None,
                   disruption = 0):
    """
    max_steps (T in the paper) - maximum number of steps to run the simulation
    """
    initial_distr = init_simulation(g, prior=prior)

    distrs = []
    coins_list = []
    mean_list = []
    std_list = []
    iters_asymptotic_learning = 0
    prior_distr = initial_distr.copy()
    distrs.append(prior_distr)

    max_steps = int(max_steps)

    # For now, static friendliness; TODO: update friendliness dynamically. (Hi Christine!)
    friendliness = friendliness_mat(g)
    init_friendliness = friendliness

    if progress is None:
        progress = make_progress()

    if task_id is None:
        task_id = progress.add_task("Simulation", total=max_steps)
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
        if log is not None:
            mean = mean_distr(posterior)
            mean_list.append(mean)
            std_list.append(std_distr(posterior, mean))
            coins_list.append(coins)
            distrs.append(posterior)
        
        # system reaches asymptotic learning when all agents reach asymptotic learning   
        largest_change = np.max(np.abs(posterior - prior_distr), axis=1)
        largest_peak = 0.01 * np.max(prior_distr, axis=1)
        if np.all(largest_change < largest_peak):
            iters_asymptotic_learning += 1
        else:
            iters_asymptotic_learning = 0

        progress.update(task_id, advance=1)

        if iters_asymptotic_learning == asymptotic_learning_max_iters:
            progress.update(task_id, total=i+1, completed=i+1)
            break

        prior_distr = posterior.copy()

        if done_event.is_set():
            print("!! Caught interrupt.")
            return

    adjacency = adjacency_mat(g)
    asymptotic = iters_asymptotic_learning == asymptotic_learning_max_iters

    if log is None:
        return SimResults(steps=steps,
                          asymptotic=asymptotic,
                          coins=None,
                          mean_list=None,
                          std_list=None,
                          final_distr=None,
                          initial_distr=None,
                          adjacency=None,
                          friendliness=None,
                          distrs=None)
    else:
        return SimResults(steps=steps,
                        asymptotic=asymptotic,
                        coins=coins_list,
                        mean_list=mean_list,
                        std_list=std_list,
                        final_distr=prior_distr,
                        initial_distr=initial_distr,
                        adjacency=adjacency,
                        friendliness=friendliness,
                        distrs=distrs)


def run_ensemble(runs: int, gen_graph, sim_params=None, title="Ensemble"):
    """
    sim_params: dictionary of parameters to pass to run_simulation
    eg: sim_params = { "max_steps": 100 }
    """
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

    coinslist = None

    with make_progress() as progress:
        results = []
        ensemble = progress.add_task(title, total=runs)

        for r in range(runs):
            task_id = progress.add_task("Sim #{}".format(r+1))

            if coinslists is not None:
                coinslist = coinslists[r]
                sim_params["prior"] = priors[r]

            sim = run_simulation(gen_graph(), coinslist=coinslist, **sim_params,
                                 task_id=task_id, progress=progress)
            results.append(sim)
            progress.update(ensemble, advance=1)

    return results



