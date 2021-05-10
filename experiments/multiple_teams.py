import datetime
import os
import random
from copy import copy, deepcopy

from agent.switching_agents import OptimalSwitching, UCRL2Switching, UCRL2MC, SwitchingAgent
from agent.agents import NoisyDriverAgent, FixedAgent, ConfidenceSet, CopyAgent

from environments.env import Environment, GridWorld
from environments.episodic_mdp import EpisodicMDP
from environments.make_envs import HEIGHT, WIDTH, make_state_extractor, FeatureStateHandler

from definitions import ROOT_DIR

from experiments.utils import test_agent_policy, test_switching_agent

import numpy as np
import pickle

from experiments.utils import evaluate_switching
from plot.plot_path import PlotPath


def run_multiple_teams_exp(env: Environment, mdp: EpisodicMDP, n_teams: int, n_episode: int, agents: list,
                           agents_cost: list, switching_cost: float, n_try: int = 1, verbose: bool = True,
                           save_agents: bool = True, time_changing: list = None, unknown_agents: list = None):
    """
    Train the switching policy in the lane driving env for multiple teams. In each episode, it generates a new
    trajectory of the lane driving env with random initial traffic level and runs UCRL2_MC, and UCRL2 algorithms to
    find the switching policy.

    Parameters
    ---------
    env: Environment

    mdp: EpisodicMDP
        The mdp in which we train the switching policy

    n_teams: number of teams

    n_episode: int

    agents: list of list
        List of list of agents, i.e., [[machine agent 1, human agent 1], ...]

    agents_cost: list
        List of agents cost, i.e., [machine cost, human cost]

    switching_cost: float
        Cost of switching

    n_try: int
        Number of repeating the experiment in each episode to evaluate the algorithms

    verbose : bool
        If `True`, then it will print logs

    save_agents: bool
        If `True`, then it saves the agent each 1000 episodes

    time_changing: list
        List of agents with time changing policy, unknown policies cannot be timechanging

    Returns
    -------
    regrets: dict
        A list of agents regret
    """

    if time_changing is None:
        time_changing = []

    if unknown_agents is None:
        unknown_agents = [i for i in range(len(agents))]

    shared_env_confidence = ConfidenceSet(shape=(mdp.n_state, mdp.n_action, mdp.n_state), delta=0.1)
    ucrl2_mc_agents = []
    ucrl2_agents = []
    optimal_agents = []

    for i in range(n_teams):
        optimal_agents.append(OptimalSwitching(mdp, agents[i], switching_cost, agents_cost, time_changing))

        ucrl2_mc_agents.append(
            UCRL2MC(mdp, agents[i], switching_cost, agents_cost, delta=0.1, unknown_agents=unknown_agents,
                    time_changing=time_changing, total_episodes=n_episode, env_prior=shared_env_confidence)

        )
        ucrl2_agents.append(
            UCRL2Switching(mdp, agents[i], switching_cost, agents_cost, delta=0.1, total_episodes=n_episode)
        )

    total_regrets = {'ucrl2_mc': [], 'ucrl2': []}
    switching_agents = {'ucrl2_mc': ucrl2_mc_agents,
                        'ucrl2': ucrl2_agents,
                        'optimal': optimal_agents}

    feat_ext = make_state_extractor(env)

    for ep in range(n_episode):
        costs_ep = {'ucrl2_mc': 0, 'ucrl2': 0, 'optimal': 0}
        for n in range(n_teams):
            # update policies
            for switching_agent in switching_agents.values():
                switching_agent[n].update_policy(ep)

            # initialize a new grid trajectory for this episode with random initial traffic level
            init_traffic = random.choice(['no-car', 'light', 'heavy'])
            grid_env = Environment().generate_grid_world(width=WIDTH, height=HEIGHT, init_traffic_level=init_traffic)

            # evaluate and train algorithms
            for name, switching_agent in switching_agents.items():
                value = evaluate_switching(switching_agent=switching_agent[n], agents_cost=agents_cost,
                                           switching_cost=switching_cost, trajectory=grid_env, agents=agents[n],
                                           feat_ext=feat_ext, n_try=n_try)
                costs_ep[name] += value

        for name in total_regrets:
            total_regrets[name].append(costs_ep[name] - costs_ep['optimal'])

        # print logs
        if verbose and ep % 10 == 0:
            results = {k: sum(v) for k, v in total_regrets.items()}
            print(f'{datetime.datetime.now()}, Episode {ep}, cumulative total regret: {results}')

        # save regrets
        if (ep % 100 == 0) and (ep // 100 > 0) and save_agents:
            with open(ROOT_DIR + '/outputs/agents/multiple_teams/total_regret', 'wb') as file:
                pickle.dump(total_regrets, file, pickle.HIGHEST_PROTOCOL)

    return switching_agents, total_regrets
