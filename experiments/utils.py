from collections import defaultdict
import random
import torch 
import numpy as np


from agent.agents import Agent
from agent.switching_agents import FixedSwitchingHuman
from environments.env import GridWorld, Environment, TYPE_COSTS
from environments.utils_env import state2features, feature2onehot
from plot.plot_path import HUMAN_COLOR, MACHINE_COLOR, PlotPath
 


def learn_evaluate(switching_agent: Agent, acting_agents, env: GridWorld,is_learn: bool, ret_trajectory=False, n_try=1, plt_path=None, machine_only=False):
    """
    Learn (on policy) or evaluate overall policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The actings agents

    env: GridWorld
        The environment for the current episode   

    is_learn: bool
        Indicates if we are training or evaluating. If `is_learn == True`,
        then `n_try = 1`, and we will update the policy for the agents

    ret_trajectory: bool
        To gather and return or not the trajectory

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """
    total_costs = 0
    
    if ret_trajectory:
        trajectory = []

    human_cf_lines = []
    human_cf_costs = []
    for i in range(n_try):       
        env.reset()
        d_tminus1 = 0
        timestep = 0
        while True:
            timestep+=1
            current_state = env.current_state()
            src = env.current_coord

            d_t = switching_agent.take_action(current_state)
            option = acting_agents[d_t] 
            if not d_t:  
               action = option.take_action(current_state, d_tminus1)
            else:
                action, policy = option.take_action(current_state)
                if plt_path is not None:
                    for key in range(len(human_cf_lines)):
                        cf_src =  human_cf_lines[key][-1][1]
                        cf_state = env.coords2state(cf_src[0], cf_src[1])
                        cf_action =  acting_agents[0].take_action(cf_state)
                        cf_dst = env.next_coords(cf_src[0], cf_src[1], cf_action)
                        human_cf_lines[key].append((cf_src, cf_dst))
                        human_cf_costs[key]+=(env.type_costs[env.cell_types[cf_dst]])

                    if (not machine_only) or (machine_only and timestep==1):# record here human alternative
                        human_only_action = acting_agents[0].take_action(current_state)
                        human_only_dst = env.next_cell(human_only_action, move=False)[0]
                        human_cf_lines.append([(src, human_only_dst)])
                        human_cf_costs.append(total_costs + env.type_costs[env.cell_types[human_only_dst]])
                            

            
            next_state, cost, finished = env.step(action)
            if finished:
                break 
            dst = env.current_coord

            c_tplus1 = cost + option.control_cost
            if ret_trajectory:
                acting_agents[0].update_policy(current_state,action)
                trajectory.append((current_state, action, next_state, cost))
            if is_learn:
                if switching_agent.trainable:
                    next_features = state2features(next_state, switching_agent.n_state_features) 
                    with torch.no_grad():
                        d_tplus1 = switching_agent.take_action(next_state)
                        if switching_agent.network.needs_agent_feature :
                            next_features.extend(feature2onehot(d_tplus1,2))
                        v_tplus1 = switching_agent.network(next_features)

                    features = state2features(current_state, switching_agent.n_state_features)
                    if switching_agent.network.needs_agent_feature :
                        features.extend(feature2onehot(d_t,2))
                    v_t = switching_agent.network(features)
                    
                    td_error = c_tplus1 + v_tplus1 - v_t
                    assert td_error
                    switching_agent.update_policy(1, td_error)
                    assert torch.any(list(switching_agent.network.parameters())[0].grad > 0.)

            
                if option.trainable and d_t:
                    delta = v_t.detach()
                    option.update_policy(d_t, delta, policy, action)

            total_costs += c_tplus1            

            if plt_path is not None:               
                clr = MACHINE_COLOR if d_t else HUMAN_COLOR
                plt_path.add_line(src, dst, clr)
    
    if human_cf_costs:
        key = np.argmin(human_cf_costs)
        for src, dst in human_cf_lines[key][:-1]:
            plt_path.add_line(src, dst, HUMAN_COLOR)
    if ret_trajectory:
        return trajectory
                    

    return total_costs / n_try


def learn_off_policy(switching_agent: Agent, acting_agents, trajectory , n_try=1, plt_path=None):
    """
    Learn  overall policy off-policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The actings agents (always contains 2)

    trajectory: list
        The trajectory induced by the behavior policy   

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """

    for i in range(n_try):
        M_t = 0
        F_t = 0
        for t in trajectory:
            (current_state, action, next_state, cost) = t            

            d_t = switching_agent.take_action(current_state)
            option = acting_agents[d_t]          
                        
            c_tplus1 = cost + option.control_cost
            
            if switching_agent.trainable:
                next_features = state2features(next_state, switching_agent.n_state_features) 
                with torch.no_grad():
                    d_tplus1 = switching_agent.take_action(next_state)
                    if switching_agent.network.needs_agent_feature :                        
                        next_features.extend(feature2onehot(d_tplus1,2))
                    v_tplus1 = switching_agent.network(next_features)

                features = state2features(current_state, switching_agent.n_state_features)
                if switching_agent.network.needs_agent_feature :
                    features.extend(feature2onehot(d_t,2))
                v_t = switching_agent.network(features)
                
                td_error = c_tplus1 + v_tplus1 - v_t

                mu_t = acting_agents[0].get_policy_approximation(current_state, action)
                
                policy = acting_agents[1].take_action(current_state)[1]
                with torch.no_grad():
                    machine_pi_t = policy.probs[action].item()
                rho = machine_pi_t / mu_t
                
                var_pi_t = machine_pi_t if d_t else mu_t
                var_rho = var_pi_t / mu_t
                F_t = 1 + var_rho * F_t

                emphatic_weighting  = rho * F_t 
                assert td_error 
                assert emphatic_weighting            
                switching_agent.update_policy(emphatic_weighting, td_error)
                assert torch.any(list(switching_agent.network.parameters())[0].grad > 0.)
        
            if acting_agents[1].trainable:
                delta = cost + v_tplus1 - v_t.detach()
                M_t = d_t + var_rho*M_t
                emphatic_weighting = rho * M_t
                assert emphatic_weighting
                assert delta
                acting_agents[1].update_policy(emphatic_weighting, delta, policy, action)
                assert torch.any(list(acting_agents[1].network.parameters())[0].grad > 0.)
    
    

            
            

    

def gather_human_trajectories(human: Agent, env_gen: Environment, n_episodes: int,**env_params):
    """ 
    Return trajectories induced by human acting alone

    Parameters
    ----------
    human: Agent
        The human agent

    env_gen: Environment
        The gridworld generator

    n_episodes: int
        The number of episodes

    env_params: dict of ('parameter_name', parameter_value)
        The gridworld parameters

    Returns
    -------
    trajectories: list of trajectory
        The list of the gathered trajectories

    """
    # width, height, init_traffic = env_params['width'], env_params['height'], env_params['init_traffic']
    trajectories = []
    fixed_switch = FixedSwitchingHuman()
    
    for ep in range(n_episodes):
        env = env_gen.generate_grid_world(**env_params)
        traj = learn_evaluate(fixed_switch, [human], env, is_learn = False, ret_trajectory=True)
        trajectories.append(traj)
    print(ep)
    return trajectories



    
