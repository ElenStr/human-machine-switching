import datetime
from collections import defaultdict
import numpy as np
import pickle

from definitions import ROOT_DIR
from experiments.utils import learn_off_policy, learn_evaluate
from agent.agents import Agent

def save_agent_cost(name, actor, critic, costs, ratios, on_off):
    with open(f'{ROOT_DIR}/results/{name}/switching_agent_'+on_off, 'wb') as file:
        pickle.dump(critic, file, pickle.HIGHEST_PROTOCOL)
    # No need to save fixed machine in fixed actor policies case
    if actor.trainable:
        with open(f'{ROOT_DIR}/results/{name}/actor_agent_'+on_off, 'wb') as file:
            pickle.dump(actor, file, pickle.HIGHEST_PROTOCOL)
    if len(costs):
        with open(f'{ROOT_DIR}/results/{name}/costs_'+on_off, 'wb') as file:
            pickle.dump(costs, file, pickle.HIGHEST_PROTOCOL)
    if len(ratios):
        with open(f'{ROOT_DIR}/results/{name}/ratios_'+on_off, 'wb') as file:
            pickle.dump(ratios, file, pickle.HIGHEST_PROTOCOL)

def evaluate(switching_agent, acting_agents, eval_set, n_try=10, plt_path=None):
    eval_costs = []
    machine_picked_ratios = []
    for grid in eval_set:
        cost, machine_picked = learn_evaluate(switching_agent, acting_agents, [grid], is_learn=False, ret_trajectory=False, n_try=n_try, plt_path=plt_path)
        eval_costs.append(cost)
        machine_picked_ratios.append(machine_picked)
    
    return np.mean(eval_costs), np.mean(machine_picked_ratios)


def train(algos, trajectories, on_line_set,
                      eval_set, eval_freq: int, save_freq: int,
                      verbose: bool = True, save_agent: bool = True, 
                      batch_size=1, eval_tries=1):
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

    on_line_set: list of Gridworld 
        The lis of gridworlds to be used in the on-line training
    
    eval_set:
        Evaluation set of environments to keep track on training progress

    eval_freq: int
        Agents' evaluation frequenncy

    save_freq: int
        Agents' and costs; saving frequency

    verbose : bool
        If `True`, then it will print logs every eval_frequency episodes

    save_agent: bool
        If `True`, then it saves the agent each save_freq episodes

    Returns
    -------
    algos: dict of 'algorithm_name' : (switching_agent, [human, machine])
        The trained agents 

    algos_costs : list
        A dictionary containing the cost of  every algorithm in each episode
    """
    algos_costs = defaultdict(lambda:[])
    machine_picked_ratios = defaultdict(lambda:[])
    if trajectories:
        ep_l = len(trajectories[0])
        trajectories = np.asarray(trajectories, dtype=object)
        batched_trajectories = np.resize(trajectories, (len(trajectories)//batch_size, batch_size, ep_l, 4) )
        for ep,traj_batch in enumerate(batched_trajectories):
            ep+=1
            for algo, agents in algos.items():
                switching_agent, acting_agents = agents
                machine = acting_agents[1]
                
                #TODO learn off policy return sth useful maybe Q ?
                learn_off_policy(switching_agent, acting_agents, np.resize(np.hstack(traj_batch), (ep_l, batch_size, 4)))

                # print log
                if verbose and ep % eval_freq == 0 and (ep // eval_freq > 0):
                    eval_cost, machine_picked = evaluate(switching_agent, acting_agents, eval_set, n_try=eval_tries)
                    print(f'{datetime.datetime.now()}, Off-policy, Episode {ep}, {algo} evaluation cost: {eval_cost}')
                    algos_costs[algo].append(eval_cost)
                    if 'switch' in algo or 'fxd' in algo:
                        machine_picked_ratios[algo].append(machine_picked) 

                # save agent
                if save_agent and (ep % save_freq == 0) and (ep // save_freq > 0):
                    save_agent_cost(algo, machine, switching_agent, algos_costs[algo],machine_picked_ratios[algo] , 'off')
                algos[algo] = (switching_agent, acting_agents)
    
    if on_line_set:
        algos_costs = defaultdict(lambda:[])
        machine_picked_ratios = defaultdict(lambda:[])
        batched_online_set = np.resize(on_line_set, (len(on_line_set)//batch_size, batch_size))        
        for ep,grid_worlds in enumerate(batched_online_set):
            
            ep+=1
            for algo, agents in algos.items():
                switching_agent, acting_agents = agents
                machine = acting_agents[1]

                learn_evaluate(switching_agent, acting_agents, grid_worlds, batch_size=batch_size, is_learn=True)

                # print log
                if verbose and ep % eval_freq == 0 and (ep // eval_freq > 0):
                    eval_cost, machine_picked = evaluate(switching_agent, acting_agents, eval_set,n_try=eval_tries)
                    print(f'{datetime.datetime.now()}, On-policy, Episode {ep}, {algo}  evaluation cost: {eval_cost}')
                    algos_costs[algo].append(eval_cost)
                    if 'switch' in algo or 'fxd' in algo:
                        machine_picked_ratios[algo].append(machine_picked) 


                # save agent
                if save_agent and (ep % save_freq == 0) and (ep // save_freq > 0):
                    save_agent_cost(algo, machine, switching_agent,algos_costs[algo],machine_picked_ratios[algo], 'on')   
                algos[algo] = (switching_agent, acting_agents)
            
    
    
    
    return algos, algos_costs
