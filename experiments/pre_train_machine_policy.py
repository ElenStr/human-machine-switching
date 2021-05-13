import datetime
import random
from copy import copy, deepcopy

from agent.agents import OptimalAgent, UCRL2, Agent, UniformDriverAgent
from environments.env import Environment, GridWorld
from environments.make_envs import HEIGHT, WIDTH

from definitions import ROOT_DIR

import numpy as np
import pickle

from experiments.utils import evaluate






def run_pre_train_machine(env: Environment, mdp: EpisodicMDP, n_episode: int, verbose: bool = True,
                          save_agent: bool = True):
    """
    Pre-train a machine policy in `no-car` subspace of the lane driving env. In each episode, 
    it generates a new trajectory of the lane driving env and runs UCRL2 to find the switching
    policy. To train the agent, we only run once in an episode. However, the evaluation is done 
    by repeating the experiment `n_try` times in each episode.

    Parameters
    ---------
    env: Environment

    mdp: EpisodicMDP
        The mdp in which we train the machine policy

    n_episode: int

    verbose : bool
        If `True`, then it will print logs every 1000 episodes

    save_agent: bool
        If `True`, then it saves the agent each 1000 episodes

    Returns
    -------
    ucrl2_regret : list
        A list containing the regret of UCRL2 algorithm in each episode
    machine_policy: Agent
        The pre-trained machine policy
    """

    optimal_agent = OptimalAgent(mdp=mdp)

    ucrl2_agent = UCRL2(env=mdp, delta=0.1, scale=0.1)
    ucrl2_regret = []

    feat_ext = make_state_extractor(env)

    for ep in range(n_episode):
        ucrl2_agent.scale = max(0.1 - (ep // (n_episode / 10)) * 0.01, 0)
        ucrl2_agent.update_policy(ep)
        
        # initialize a new grid trajectory for this episode with `no-car` initial traffic level
        grid_env = Environment().generate_grid_world(width=WIDTH, height=HEIGHT, init_traffic_level='no-car')

        # evaluate the optimal policy
        optimal_value = evaluate(agent=optimal_agent, env=grid_env, feat_ext=feat_ext)

        # evaluate and train ucrl2
        value = evaluate(agent=ucrl2_agent, env=grid_env, feat_ext=feat_ext)
        ucrl2_regret.append(value - optimal_value)

        # print log
        if verbose and ep % 1000 == 0:
            print(f'{datetime.datetime.now()}, Episode {ep}, UCRL2 algorithm cumulative regret: {np.sum(ucrl2_regret)}')

        # save agent
        if save_agent and (ep % 1000 == 0) and (ep // 1000 > 0):
            with open(ROOT_DIR + '/outputs/agents/pre_trained_machine/machine_agent', 'wb') as file:
                pickle.dump(ucrl2_agent, file, pickle.HIGHEST_PROTOCOL)

            with open(ROOT_DIR + '/outputs/agents/pre_trained_machine/machine_regret', 'wb') as file:
                pickle.dump(ucrl2_regret, file, pickle.HIGHEST_PROTOCOL)

    ucrl2_agent.final_update()
    with open(ROOT_DIR + '/outputs/agents/pre_trained_machine/machine_agent', 'wb') as file:
        pickle.dump(ucrl2_agent, file, pickle.HIGHEST_PROTOCOL)
    return ucrl2_agent, ucrl2_regret
