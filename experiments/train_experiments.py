import datetime
from collections import defaultdict
import numpy as np
import pickle

from definitions import ROOT_DIR
from experiments.utils import learn_off_policy, learn_evaluate
from agent.agents import Agent

def save_agent_cost(name, actor, critic, costs, on_off):
    with open(f'{ROOT_DIR}/outputs/agents/{name}/switching_agent_'+on_off, 'wb') as file:
        pickle.dump(critic, file, pickle.HIGHEST_PROTOCOL)
    # No need to save fixed machine in fixed actor policies case
    if actor.trainable:
        with open(f'{ROOT_DIR}/outputs/agents/{name}/actor_agent_'+on_off, 'wb') as file:
            pickle.dump(actor, file, pickle.HIGHEST_PROTOCOL)
    if len(costs):
        with open(f'{ROOT_DIR}/outputs/agents/{name}/costs_'+on_off, 'wb') as file:
            pickle.dump(costs, file, pickle.HIGHEST_PROTOCOL)


def train_experiments(algos, trajectories, env_generator, n_episode_on: int,
                          verbose: bool = True, save_agent: bool = True):
    """
    Train the switching and machine policy for different configurations
    of machine and switching agents.

    Parameters
    ---------
    algos: dict of 'algorithm_name' : (switching_agent, [human, machine])
        The switching agents and the acting agents to be trained

    trajecotries: List of List of tuples 
        The trajectories induced by the human acting alone, 
        needed for the off-policy stage.

    env_generator: lambda: environment.generate_gridworld(args)
        The gridworld generator for the on-policy stage
    
    n_episode_on: int
        Number of episodes in the on-policy stage

    verbose : bool
        If `True`, then it will print logs every 1000 episodes

    save_agent: bool
        If `True`, then it saves the agent each 1000 episodes

    Returns
    -------
    algos: dict of 'algorithm_name' : (switching_agent, [human, machine])
        The trained agents 

    algos_costs : list
        A dictionary containing the cost of  every algorithm in each episode
    """
    algos_costs = defaultdict(lambda:[])


    for ep,traj in enumerate(trajectories):
        for algo, agents in algos.items():
            switching_agent, acting_agents = agents
            machine = acting_agents[1]
            
            #TODO learn off policy return sth useful maybe Q ?
            learn_off_policy(switching_agent, acting_agents, traj)
            # algos_costs.append(ep_cost)

            # # print log
            # if verbose and ep % 1000 == 0 and (ep // 1000 > 0):
            #     print(f'{datetime.datetime.now()}, Episode {ep}, Fully automated off-policy algorithm cumulative cost: {np.sum(algos_costs)}')

            # save agent
            if save_agent and (ep % 1000 == 0) and (ep // 1000 > 0):
                save_agent_cost(algo, switching_agent, machine, algos_costs[algo], 'off')
        
    for ep in range(n_episode_on):
        grid_world = env_generator()
        for algo, agents in algos.items():
            switching_agent, acting_agents = agents
            machine = acting_agents[1]

            ep_cost = learn_evaluate(switching_agent, acting_agents, grid_world, is_learn=True)
            algos_costs[algo].append(ep_cost)

            # print log
            if verbose and ep % 1000 == 0 and (ep // 1000 > 0):
                print(f'{datetime.datetime.now()}, Episode {ep}, {algo} on-policy training \
                            average cost per 1000: {np.sum(algos_costs[algo][-1000:])/1000}')

            # save agent
            if save_agent and (ep % 1000 == 0) and (ep // 1000 > 0):
                save_agent_cost(algo, switching_agent, machine, algos_costs[algo], 'on')

    
    
    
    
    return algos, algos_costs
