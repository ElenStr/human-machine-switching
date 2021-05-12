import random
import torch 

from agent.agents import Agent
from environments.env import GridWorld
from environments.utils_env import WIDTH, HEIGHT, state2features
from plot.plot_path import HUMAN_COLOR, MACHINE_COLOR, PlotPath
 


def learn_evaluate(switching_agent: Agent, human: Agent, 
                   machine: Agent, env: GridWorld, learn_actor: bool, 
                   learn_critic: bool,n_try=1, plt_path=None):
    """
    Learn (on policy) or evaluate switching policies in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    human:  Agent
        The human agent

    machine:  Agent
        The machine agent 

    env: GridWorld
        The environment for the current episode   

    learn_actor : bool
        Indicates if we are training or evaluating. If `is_learn == True`,
        then `n_try = 1`, and we will update the policy for the learning actor

    learn_critic : bool
        Indicates if we are training or evaluating. If `is_learn == True`,
        then `n_try = 1`, and we will update the policy for the learning critic

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """
    total_costs = 0
    count = 0

    for i in range(n_try):
        count += 1
        finished = False
        env.reset()
        while not finished:
            current_state = env.current_state()
            src = env.current_coord
            d_s = switching_agent.take_action(current_state)
            option = machine if d_s else human
            action,policy = option.take_action(current_state)[0] 
            
            next_state, cost, finished = env.step(action)
            dst = env.current_coord

            c_tplus1 = cost + option.control_cost
            if learn_critic:
                next_features = state2features(next_state) 
                with torch.no_grad():
                    d_tplus1 = switching_agent.take_action(next_state)
                    if switching_agent.network.needs_agent_feature :
                        next_features.append(d_tplus1 + 1.)
                    v_tplus1 = switching_agent.network(next_features)
                features = state2features(current_state)
                if switching_agent.network.needs_agent_feature :
                    features.append(d_s + 1.)
                v_t = switching_agent.network(features)
                td_error = c_tplus1 + v_tplus1 - v_t
                switching_agent.update_policy(1, td_error)
            
            if learn_actor and d_s:
                delta = v_t.detach()
                option.update_policy(d_s, delta, policy)





            if not finished:
                total_costs += c_tplus1

            

            if plt_path is not None and not finished:
                clr = MACHINE_COLOR if d_s else HUMAN_COLOR
                plt_path.add_line(src, dst, clr)

            
    return total_costs / count




