import datetime
import os
import random
from copy import copy, deepcopy

from agent.switching_agents import OptimalSwitching, UCRL2Switching, UCRL2MC, SwitchingAgent
from agent.agents import NoisyDriverAgent, FixedAgent, ConfidenceSet, CopyAgent

from environments.env import Environment, GridWorld
from environments.episodic_mdp import EpisodicMDP
# from environments.utils_env import HEIGHT, WIDTH

from definitions import ROOT_DIR

from experiments.utils import test_agent_policy, test_switching_agent

import numpy as np
import pickle

from experiments.utils import evaluate_switching
from plot.plot_path import PlotPath



def run_single_team_exp(env: Environment, mdp: EpisodicMDP, n_episode: int, agents: list, agents_cost: list,
                        switching_cost: float, n_try: int = 1, verbose: bool = True, save_dir: str = None,
                        plot_epochs: list = None, save_agents: bool = True, plot_dir: str = None,
                        time_changing: list = None, unknown_agents: list = None):
    """
    Train the switching policy in the lane driving env. In each episode, it generates a new trajectory of the lane
    driving env with random initial traffic level and runs UCRL2_MC, UCRL2, human-only and machine-only algorithms to
    find the switching policy.

    Parameters
    ---------
    env: Environment

    mdp: EpisodicMDP
        The mdp in which we train the switching policy

    n_episode: int

    agents: list
        List of agents, i.e., [machine agent, human agent]

    agents_cost: list
        List of agents cost, i.e., [machine cost, human cost]

    switching_cost: float
        Cost of switching

    env_prior: ConfidenceSet
        Prior confidence set over environment transitions, can be only used in UCRL2_MC

    n_try: int
        Number of repeating the experiment in each episode to evaluate the algorithms

    verbose : bool
        If `True`, then it will print logs

    plot_epochs: list
        List of episode numbers in which the trajectory induced by UCRL2_MC will be plotted

    save_agents: bool
        If `True`, then it saves the agent each 1000 episodes

    plot_dir: str
        Directory of epoch plots

    time_changing: list
        List of agents with time changing policy, unknown policies cannot be timechanging

    Returns
    -------
    agents : dict
        A list of trained switching agents [ucrl2_mc, ucrl2, machine, human]

    regrets: dict
        A list of agents regret
    """
    if plot_epochs is None:
        plot_epochs = []

    if time_changing is None:
        time_changing = []

    if unknown_agents is None:
        unknown_agents = [i for i in range(len(agents))]

    optimal_switching_agent = OptimalSwitching(mdp, agents, switching_cost, agents_cost, time_changing)

    ucrl2_mc_agent = UCRL2MC(mdp, agents, switching_cost, agents_cost, delta=0.1, unknown_agents=unknown_agents,
                             time_changing=time_changing, total_episodes=n_episode)

    ucrl2_agent = UCRL2Switching(mdp, agents, switching_cost, agents_cost, delta=0.1, total_episodes=n_episode)

    human_only_agent = FixedAgent(mdp, 1)
    machine_only_agent = FixedAgent(mdp, 0)

    regrets = {
               'ucrl2_mc': [],
               'ucrl2': [],
               'human_only': [],
               'machine_only': []}

    switching_agents = {
                        'ucrl2_mc': ucrl2_mc_agent,
                        'ucrl2': ucrl2_agent,
                        'human_only': human_only_agent,
                        'machine_only': machine_only_agent,
                        'optimal': optimal_switching_agent}

    feat_ext = make_state_extractor(env)

    for ep in range(n_episode):

        # update policies
        for switching_agent in switching_agents.values():
            switching_agent.update_policy(ep)

        # initialize a new grid trajectory for this episode with random initial traffic level
        init_traffic = random.choice(['no-car', 'light', 'heavy'])
        grid_env = Environment().generate_grid_world(width=WIDTH, height=HEIGHT, init_traffic_level=init_traffic)

        plt_path = None
        costs = {}
        # evaluate and train algorithms
        for name, switching_agent in switching_agents.items():
            plt_path = None
            if ep in plot_epochs and isinstance(switching_agent, UCRL2MC):
                plt_path = PlotPath(grid_env, n_try)
            value = evaluate_switching(switching_agent=switching_agent, agents_cost=agents_cost,
                                       switching_cost=switching_cost, trajectory=grid_env, agents=agents,
                                       feat_ext=feat_ext, n_try=n_try, plt_path=plt_path)
            costs[name] = value
            if plt_path is not None:
                plot_name = f'{name}_ep_{ep}.png'
                plot_file = os.path.join(plot_dir, plot_name)
                plt_path.plot(plot_file)

        for name in regrets:
            regrets[name].append(costs[name] - costs['optimal'])

        # print logs
        if verbose and ep % 500 == 0:
            results = {k: sum(v) for k, v in regrets.items()}
            print(f'{datetime.datetime.now()}, Episode {ep},\n'
                  f'cumulative costs: {results}\n')

        # save agents and regrets
        if (ep % 1000 == 0) and (ep // 1000 > 0) and save_agents:
            with open(ROOT_DIR + '/outputs/agents/single_team/switching_agents', 'wb') as file:
                pickle.dump(switching_agents, file, pickle.HIGHEST_PROTOCOL)

            with open(ROOT_DIR + '/outputs/agents/single_team/switching_regrets', 'wb') as file:
                pickle.dump(regrets, file, pickle.HIGHEST_PROTOCOL)

    return switching_agents, regrets