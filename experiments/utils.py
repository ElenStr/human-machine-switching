from collections import defaultdict
import random
import torch 
import numpy as np


from agent.agents import Agent
from agent.switching_agents import FixedSwitchingHuman
from environments.env import GridWorld, Environment, TYPE_COSTS
from environments.utils_env import state2features, feature2onehot
from plot.plot_path import HUMAN_COLOR, MACHINE_COLOR, PlotPath
 


def learn_evaluate(switching_agent: Agent, acting_agents, env: GridWorld,is_learn: bool, ret_trajectory=False, n_try=1, plt_path=None, machine_only=False, not_batch=True):
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

            d_t = switching_agent.take_action(current_state, is_learn)
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
                            
            d_tminus1 = d_t
            
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
                        d_tplus1 = switching_agent.take_action(next_state,is_learn)
                        if switching_agent.network.needs_agent_feature :
                            next_features.extend(feature2onehot(d_tplus1,2))
                        v_tplus1 = switching_agent.target_network(next_features)

                    features = state2features(current_state, switching_agent.n_state_features)
                    if switching_agent.network.needs_agent_feature :
                        features.extend(feature2onehot(d_t,2))
                    v_t = switching_agent.network(features)
                    
                    td_error = c_tplus1 + v_tplus1 - v_t
                    if td_error==0.:
                        print('TD Error')
                        print(c_tplus1,v_tplus1,v_t )
                    # if td_error != 0. :
                    switching_agent.update_policy(1, td_error,len(next_state)==1 or not_batch)
                    if torch.is_tensor(list(switching_agent.network.parameters())[0].grad):
                        if not torch.any(list(switching_agent.network.parameters())[0].grad > 0.):
                            print('critic zero grad ')
                        if not torch.all(list(switching_agent.network.parameters())[-1].grad < 1e3):
                            print('critic grad > 1e3')


            
                if option.trainable:
                    # print(policy.log_prob(policy.sample()).exp())
                    with torch.no_grad():
                        v_tplus1 = switching_agent.target_network(next_features)
                        v_t = switching_agent.network(features)
                        # d_t = switching_agent.take_action(current_state)
                
                    delta = cost + v_tplus1 - v_t
                    if not delta:
                        print('delta ',v_tplus1,v_t)
                    
                    # if delta !=0. and d_t==1:
                    option.update_policy(d_t, delta, policy, action, len(next_state)==1 or not_batch)
                    
                    if torch.is_tensor(list(option.network.parameters())[0].grad):
                        if not torch.any(list(option.network.parameters())[0].grad > 0.):
                            print('actor zero grad ')
                        if not torch.all(list(option.network.parameters())[-1].grad < 1e3):
                            print('actor grad > 1e3')



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


def learn_off_policy(switching_agent: Agent, acting_agents, trajectory , n_try=1, plt_path=None, not_batch=True):
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
        
        for t in trajectory:
            (current_state, action, next_state, cost) = t            

            d_t = switching_agent.take_action(current_state, True)
            option = acting_agents[d_t]          
                        
            c_tplus1 = cost + option.control_cost
            
            if switching_agent.trainable:
                next_features = state2features(next_state, switching_agent.n_state_features) 
                with torch.no_grad():
                    d_tplus1 = switching_agent.take_action(next_state, True)
                    if switching_agent.network.needs_agent_feature :                        
                        next_features.extend(feature2onehot(d_tplus1,2))
                    v_tplus1 = switching_agent.target_network(next_features)

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

                var_rho_prev = switching_agent.var_rho
                switching_agent.F_t = 1 + var_rho_prev * switching_agent.F_t

                var_pi_t = machine_pi_t if d_t else mu_t
                switching_agent.var_rho = var_pi_t / mu_t

                emphatic_weighting  = rho * switching_agent.F_t 
                
                if not td_error:
                    print("TD error")
                    print(c_tplus1, v_tplus1, v_t)  
                    print(current_state, action, next_state)                 
                    

                if not emphatic_weighting:
                    print('critic emphatic')
                    print(rho, switching_agent.F_t, var_rho_prev)

                # batched update no problem for td = 0
                switching_agent.update_policy(emphatic_weighting, td_error, len(next_state)==1 or not_batch)
                if torch.is_tensor(list(switching_agent.network.parameters())[0].grad):
                    if not torch.any(list(switching_agent.network.parameters())[0].grad > 0.):
                        print('critic zero grad ')
                    if not torch.all(list(switching_agent.network.parameters())[-1].grad < 1e3):
                        print('critic grad > 1e3')
        
            if acting_agents[1].trainable:
                with torch.no_grad():
                    v_tplus1 = switching_agent.target_network(next_features)
                    v_t = switching_agent.network(features)
                    # d_t = switching_agent.take_action(current_state)
                
                delta = cost + v_tplus1 - v_t
                # updated dt ?
                acting_agents[1].M_t = d_t + var_rho_prev*acting_agents[1].M_t
                emphatic_weighting = rho * acting_agents[1].M_t
                
                if not emphatic_weighting:
                    print('actor emphatic ',rho, var_rho_prev, acting_agents[1].M_t )
                if not delta:
                    print('delta ',cost, v_tplus1, v_t)
                    print(current_state, action, next_state)                 


                # if delta!=0. and emphatic_weighting!=0.:
                acting_agents[1].update_policy(emphatic_weighting, delta, policy, action, len(next_state)==1 or not_batch)
                
                if torch.is_tensor(list(acting_agents[1].network.parameters())[0].grad):
                    if not torch.any(list(acting_agents[1].network.parameters())[0].grad > 0.):
                        print('actor zero grad')
                    if not torch.all(list(acting_agents[1].network.parameters())[-1].grad < 1e3):
                        print('actor grad > 1e3')
    
    

            
            

    

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
    return trajectories



    
