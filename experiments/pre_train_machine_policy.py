import datetime
import random
import numpy as np
import pickle

from definitions import ROOT_DIR
from experiments.utils import learn_off_policy, learn_evaluate
from agent.agents import Agent

def save_agent_cost(agent, costs, on_off):
    with open(ROOT_DIR + '/outputs/agents/pre_trained_machine/machine_agent_'+on_off, 'wb') as file:
        pickle.dump(agent, file, pickle.HIGHEST_PROTOCOL)

    with open(ROOT_DIR + '/outputs/agents/pre_trained_machine/machine_costs_'+on_off, 'wb') as file:
        pickle.dump(costs, file, pickle.HIGHEST_PROTOCOL)


def run_pre_train_machine(switching_agent: Agent, agents,
                          trajectories, env_generator, n_episode_on: int,
                          verbose: bool = True, save_agent: bool = True):
    """
    Train the machine policy for fully automated algorithm and the fixed 
    agent policies case

    Parameters
    ---------
    switching_agent: Agent
        The fixed switching agent for machine

    agents: List of Agent 
        The machine agent to be trained, and the human agent
        that acted for the trajectories gathering (never to take action
        just needed for the human policy approximation).

    trajecotries: List of List of tuples 
        The trajectories induced by the human acting alone, 
        needed for the off-policy stage.

    env_generator: lambda: environment.generate_gridworld(args)
        The gridworld generator for the on-policy stage
    
    n_episode_on: int
        Number of episodes in the on policy stage

    verbose : bool
        If `True`, then it will print logs every 1000 episodes

    save_agent: bool
        If `True`, then it saves the agent each 1000 episodes

    Returns
    -------
    (switching, machine): tuple of Agent
        The pre-trained machine agent 

    machine_cost : list
        A list containing the cost of the machine agent in each episode
    """
    machine_costs = []
    machine = agents[1]


    for ep,traj in enumerate(trajectories):
        ep_cost = learn_off_policy(switching_agent, agents, traj)
        machine_costs.append(ep_cost)

        # print log
        if verbose and ep % 1000 == 0 and (ep // 1000 > 0):
            print(f'{datetime.datetime.now()}, Episode {ep}, Fully automated off-policy algorithm cumulative cost: {np.sum(machine_costs)}')

        # save agent
        if save_agent and (ep % 1000 == 0) and (ep // 1000 > 0):
            save_agent_cost(machine, machine_costs, 'off')
        
    for ep in range(n_episode_on):
        grid_world = env_generator()
        ep_cost = learn_evaluate(switching_agent, agents, grid_world, is_learn=True)
        machine_costs.append(ep_cost)

        # print log
        if verbose and ep % 1000 == 0 and (ep // 1000 > 0):
            print(f'{datetime.datetime.now()}, Episode {ep}, Fully automated on-policy algorithm cumulative cost: {np.sum(machine_costs)}')

        # save agent
        if save_agent and (ep % 1000 == 0) and (ep // 1000 > 0):
            save_agent_cost(machine, machine_costs, 'on')

    
    
    
    switching_agent.trainable = False
    machine.trainable = False
    # make sure machine changes
    return switching_agent, agents, machine_costs
