from collections import defaultdict
import random
import torch 
import numpy as np


from agent.agents import Agent
from agent.switching_agents import FixedSwitchingHuman
from environments.env import GridWorld, Environment, TYPE_COSTS
from environments.utils_env import state2features, feature2onehot
from plot.plot_path import HUMAN_COLOR, MACHINE_COLOR, PlotPath
 


def learn_evaluate(switching_agent: Agent, acting_agents, envs ,is_learn: bool, ret_trajectory=False, n_try=1, plt_path=None, machine_only=False,batch_size=1):
    """
    Learn (on policy) or evaluate overall policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The actings agents

    envs: list of GridWorld
        If in learning mode it contains the environment batch, 
        otherwise the environment for evaluation.

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
    total_costs = []
    
    if ret_trajectory:
        trajectory = []

    human_cf_lines = []
    human_cf_costs = []
    
    for i in range(n_try):       
        for env in envs:
            env.reset()
        d_tminus1 = np.zeros(batch_size)
        timestep = 0
        trajectory_cost = 0
        while True:
            timestep+=1
            td_errors = []
            log_pis = []
            entropies = []
            costs_for_delta = []
            v_tplus1_inp = []
            v_t_inp = []
            for b,env in enumerate(envs):
                current_state = env.current_state()
                src = env.current_coord

                d_t = switching_agent.take_action(current_state, is_learn)
                option = acting_agents[d_t] 
                if not d_t:  
                    action = option.take_action(current_state, d_tminus1[b])
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
                                
                d_tminus1[b] = d_t
                
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
                        if td_error != 0 :
                            td_errors.append(td_error)
                        if option.trainable and d_t:
                            costs_for_delta.append(cost)
                            v_tplus1_inp.append(next_features)
                            v_t_inp.append(features)

                            log_pis.append(policy.log_prob(torch.as_tensor(action)))
                            entropies.append(policy.entropy().mean())
                        
            
            if finished:
                break
            if is_learn and  switching_agent.trainable and len(td_errors):
                td_errors = torch.stack(td_errors)
                switching_agent.update_policy(1, td_errors)
                if torch.is_tensor(list(switching_agent.network.parameters())[0].grad):
                    if not torch.any(list(switching_agent.network.parameters())[0].grad > 0.):
                        print('critic zero grad ')
                    if not torch.all(list(switching_agent.network.parameters())[-1].grad < 1e3):
                        print('critic grad > 1e3')

            
            if is_learn and acting_agents[1].trainable and len(costs_for_delta):
                # print(policy.log_prob(policy.sample()).exp())
                with torch.no_grad():
                    v_tplus1 = switching_agent.target_network(v_tplus1_inp)
                    v_t = switching_agent.network(v_t_inp)
                    # d_t = switching_agent.take_action(current_state)
            
                deltas = torch.as_tensor(costs_for_delta) + v_tplus1 - v_t
                # if not delta:
                #     print('delta ',v_tplus1,v_t)
                
                # if delta !=0. and d_t==1:
                log_pis = torch.stack(log_pis)
                entropies = torch.stack(entropies)
                acting_agents[1].update_policy(1, deltas, log_pis, entropies)
                
                if torch.is_tensor(list(acting_agents[1].network.parameters())[0].grad):
                    if not torch.any(list(acting_agents[1].network.parameters())[0].grad > 0.):
                        print('actor zero grad ')
                    if not torch.all(list(acting_agents[1].network.parameters())[-1].grad < 1e3):
                        print('actor grad > 1e3')



            trajectory_cost += c_tplus1            

            if plt_path is not None:               
                clr = MACHINE_COLOR if d_t else HUMAN_COLOR
                plt_path.add_line(src, dst, clr)
        total_costs.append(trajectory_cost)
    if human_cf_costs:
        key = np.argmin(human_cf_costs)
        for src, dst in human_cf_lines[key][:-1]:
            plt_path.add_line(src, dst, HUMAN_COLOR)
    if ret_trajectory:
        return trajectory
                    

    return np.mean(total_costs)


def learn_off_policy(switching_agent: Agent, acting_agents, trajectory_batch , n_try=1, plt_path=None):
    """
    Learn  overall policy off-policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The actings agents (always contains 2)

    trajectory_batch: array 
        The trajectory batch induced by the behavior policy. 
        

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """

    for i in range(n_try):

        switching_agent.F_t = np.zeros(trajectory_batch[0].shape[0])
        acting_agents[1].M_t = np.zeros(trajectory_batch[0].shape[0])
        for t_batch in trajectory_batch:
            critic_emphatic_weightings = []
            td_errors = []
            actor_emphatic_weightings =[]
            deltas = []
            log_pis = []
            entropies = []
            costs_for_delta = []
            v_tplus1_inp = []
            v_t_inp = []
            for b,t in enumerate(t_batch):
                # print(t)

                current_state, action, next_state, cost = t            

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

                        var_rho_prev = switching_agent.var_rho[b]
                        switching_agent.F_t[b] = 1 + var_rho_prev * switching_agent.F_t[b]

                        var_pi_t = machine_pi_t if d_t else mu_t
                        switching_agent.var_rho[b] = var_pi_t / mu_t

                    emphatic_weighting  = rho * switching_agent.F_t[b] 
                    
                    if not td_error:
                        print("TD error")
                        print(c_tplus1, v_tplus1, v_t)  
                        print(current_state, action, next_state)                 
                        

                    if not emphatic_weighting:
                        print('critic emphatic')
                        print(rho, switching_agent.F_t[b], var_rho_prev)

                    # batched update no problem for td = 0
                    if emphatic_weighting != 0. and td_error !=0 :
                        critic_emphatic_weightings.append(emphatic_weighting)
                        td_errors.append(td_error)
                
                if acting_agents[1].trainable and d_t*acting_agents[1].M_t[b] !=0 :                
                    acting_agents[1].M_t[b] = d_t + var_rho_prev*acting_agents[1].M_t[b]
                    emphatic_weighting = rho * acting_agents[1].M_t[b]
                    if not emphatic_weighting:
                        print('actor emphatic ',rho, var_rho_prev, acting_agents[1].M_t[b] )
                    actor_emphatic_weightings.append(emphatic_weighting)
                    costs_for_delta.append(cost)
                    v_tplus1_inp.append(next_features)
                    v_t_inp.append(features)

                    log_pis.append(policy.log_prob(torch.as_tensor(action)))
                    entropies.append(policy.entropy().mean())

            if switching_agent.trainable and len(td_errors):
                critic_emphatic_weightings = torch.as_tensor(critic_emphatic_weightings)                
                td_errors = torch.stack(td_errors)
                switching_agent.update_policy(critic_emphatic_weightings, td_errors)

                if torch.is_tensor(list(switching_agent.network.parameters())[0].grad):
                    if not torch.any(list(switching_agent.network.parameters())[0].grad > 0.):
                        print('critic zero grad ')
                    if not torch.all(list(switching_agent.network.parameters())[-1].grad < 1e3):
                        print('critic grad > 1e3')
            if acting_agents[1].trainable and len(actor_emphatic_weightings):
                
                with torch.no_grad():
                    v_tplus1 = switching_agent.target_network(v_tplus1_inp)
                    v_t = switching_agent.network(v_t_inp)
                        # d_t = switching_agent.take_action(current_state)
                    
                deltas = torch.as_tensor(costs_for_delta) + v_tplus1 - v_t
                    # updated dt ?                   
                if not deltas.any():
                    print('Deltas ') #,cost, v_tplus1, v_t)
                    # print(current_state, action, next_state)                 
                    
                    

                    # if delta!=0. and emphatic_weighting!=0.:
                actor_emphatic_weightings = torch.as_tensor(actor_emphatic_weightings)
                # deltas = torch.as_tensor(deltas)
                log_pis = torch.stack(log_pis)
                entropies = torch.stack(entropies)
                acting_agents[1].update_policy(actor_emphatic_weightings, deltas, log_pis, entropies)
                
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
        traj = learn_evaluate(fixed_switch, [human], [env], is_learn = False, ret_trajectory=True)
        trajectories.append(traj)
    return trajectories



    
