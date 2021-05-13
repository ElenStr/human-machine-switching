import random
import torch 
from collections import deque
import numpy as np

from agent.agents import Agent
from environments.env import GridWorld
from environments.utils_env import state2features
from plot.plot_path import HUMAN_COLOR, MACHINE_COLOR, PlotPath
 


def learn_evaluate(switching_agent: Agent, acting_agents, env: GridWorld,is_learn: bool, ret_trajectory=False, n_try=1, plt_path=None):
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
    count = 0
    if ret_trajectory:
        trajectory = deque()


    for i in range(n_try):
        count += 1
        finished = False
        env.reset()
        d_tminus1 = 0
        while not finished:
            current_state = env.current_state()
            src = env.current_coord

            d_t = switching_agent.take_action(current_state)
            option = acting_agents[d_t] 
            if not dt:  
               action = option.take_action(current_state, d_tminus1)
            else:
                action, policy = option.take_action(current_state)
            
            next_state, cost, finished = env.step(action)
            dst = env.current_coord

            c_tplus1 = cost + option.control_cost
            if ret_trajectory:
                trajectory.append((current_state, action, next_state, cost))
            if is_learn:
                if switching_agent.trainable:
                    next_features = state2features(next_state) 
                    with torch.no_grad():
                        d_tplus1 = switching_agent.take_action(next_state)
                        if switching_agent.network.needs_agent_feature :
                            next_features.append(d_tplus1 + 1.)
                        v_tplus1 = switching_agent.network(next_features)

                    features = state2features(current_state)
                    if switching_agent.network.needs_agent_feature :
                        features.append(d_t + 1.)
                    v_t = switching_agent.network(features)
                    
                    td_error = c_tplus1 + v_tplus1 - v_t
                    switching_agent.update_policy(1, td_error)
            
                if option.trainable and d_t:
                    delta = v_t.detach()
                    option.update_policy(d_t, delta, policy, action)

            if not finished:
                total_costs += c_tplus1            

            if plt_path is not None and not finished:
                clr = MACHINE_COLOR if d_t else HUMAN_COLOR
                plt_path.add_line(src, dst, clr)

    if ret_trajectory:
        return trajectory

    return total_costs / count

def learn_off_policy(switching_agent: Agent, acting_agents, trajectory , n_try=1, plt_path=None):
    """
    Learn  overall policy off-policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The actings agents (always contains 2)

    trajectory: deque
        The trajectory induced by the behavior policy   

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """
    total_costs = 0
    count = 0
     

    for i in range(n_try):
        count += 1
        M_t = 0
        F_t = 0
        while trajectory :
            (current_state, action, next_state, cost) = trajectory.popleft()            

            d_t = switching_agent.take_action(current_state)
            option = acting_agents[d_t]           
            

            c_tplus1 = cost + option.control_cost
            
            if switching_agent.trainable:
                next_features = state2features(next_state) 
                with torch.no_grad():
                    d_tplus1 = switching_agent.take_action(next_state)
                    if switching_agent.network.needs_agent_feature :
                        next_features.append(d_tplus1 + 1.)
                    v_tplus1 = switching_agent.network(next_features)

                features = state2features(current_state)
                if switching_agent.network.needs_agent_feature :
                    features.append(d_t + 1.)
                v_t = switching_agent.network(features)
                
                td_error = c_tplus1 + v_tplus1 - v_t
                mu_t = acting_agents[0].get_policy_approximation(current_state, action)
                
                policy = acting_agents[1].take_action(next_state)[1]
                machine_pi_t = policy.detach().probs[action].item()
                rho = machine_pi_t / mu_t
                
                var_pi_t = machine_pi_t if d_t else mu_t
                var_rho = var_pi_t / mu_t
                F_t = 1 + var_rho * F_t

                emphatic_weighting  = rho * F_t              
                switching_agent.update_policy(emphatic_weighting, td_error)
        
            if acting_agents[1].trainable:
                delta = cost + v_tplus1 - v_t.detach()
                M_t = d_t + var_rho*M_t
                emphatic_weighting = rho * M_t
                option.update_policy(emphatic_weighting, delta, policy, action)

            
            total_costs += c_tplus1            


    return total_costs / count

    



