import torch 
import numpy as np
import sys

from agent.agents import Agent
from agent.switching_agents import FixedSwitchingHuman, FixedSwitchingMachine
from environments.env import Environment
from environments.utils_env import *
from config import TRIPS, ENV

 

def learn_evaluate(switching_agent: Agent, acting_agents, trip_id ,is_learn: bool, online_ev=False, grid_id=0,ret_trajectory=False,  n_try=1, batch_size=1):
    """
    Learn (on policy) or evaluate overall policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The actings agents

    trip_id: int
        If in learning mode it contains the trip_id to train for, 
        otherwise the trip for evaluation.

    is_learn: bool
        Indicates if we are training or evaluating. If `is_learn == True`,
        then `n_try = 1`, and we will update the policy for the agents

    ret_trajectory: bool
        To gather and return or not the trajectory/trajectories.
    
    online_ev: bool
        True in online evaluation

    grid_id: int (unused for now)
        Used only if ret_trajectory==True, the unique grid id for which 
        human policy distribution approximation is computed if applied

    Returns
    -------

    np.mean(total_costs), np.mean(total_machine_picked) : float,float
        Average total cost of the trajectory and average time machine was picked

    trajectories: list of trajectory
        The recorded trajectory/-ies while acting in the graph
    """
    total_costs = []
    total_machine_picked =[]
    
    if ret_trajectory:
        trajectories = []
    # Needed s.t. method machine can ingore obstacles (see paper Scenarios I,III)    
    # ignore_obstacle= ''
    # if len(acting_agents) > 1 and (acting_agents[1].setting == 2 or acting_agents[1].setting == 7) and isinstance(switching_agent, FixedSwitchingMachine):
        # ignore_obstacle= obstacle_to_ignore
    
    for i in range(n_try):
        # TODO: now works only with batch = 1 if ret_trajectory
        if ret_trajectory:
            trajectory = []       
        # for env in envs:
            # env.reset()
        start_id = TRIPS[trip_id][0]
        finish_id = TRIPS[trip_id][1]

        ENV.reset(start_id, finish_id)

        d_tminus1 = 0
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
            d_ts = []

            current_state = ENV.current_state()
            

            d_t = switching_agent.take_action(current_state, train=is_learn, online=online_ev)
            
            option = acting_agents[d_t] 
            total_machine_picked.append(d_t)
            if not d_t:  
                action = option.take_action(current_state, d_tminus1)
            else:
                action, policy = option.take_action(current_state)
                            
            d_tminus1 = d_t
            
            next_state, cost, finished = ENV.step(action)
            if finished:
                break 
            

            c_tplus1 = cost + option.control_cost
            # unused for now
            if ret_trajectory:
                if not acting_agents[0].actual:
                    acting_agents[0].update_policy(current_state,action, grid_id)
                trajectory.append((current_state, action, next_state, cost, grid_id))
            
            if is_learn:
                if switching_agent.trainable:                       
                    next_features = Environment.state2features(next_state, switching_agent.n_state_features) 
                    with torch.no_grad():
                        d_tplus1 = switching_agent.take_action(next_state,train=is_learn, use_target=True)
                        if switching_agent.network.needs_agent_feature :
                            next_features = [*next_features, *Environment.agent_feature2net_input(d_tplus1)]
                        v_tplus1 = switching_agent.target_network(next_features)

                    features = Environment.state2features(current_state, switching_agent.n_state_features)
                    if switching_agent.network.needs_agent_feature :
                        features = [*features, *Environment.agent_feature2net_input(d_t)]
                    v_t = switching_agent.network(features)
                    
                    td_error = c_tplus1 + v_tplus1 - v_t
                    if td_error==0.:
                        print('TD Error')
                        print(c_tplus1,v_tplus1,v_t )

                    td_errors.append(td_error)
                    
                    if option.trainable:
                        
                        costs_for_delta.append(cost)
                        v_tplus1_inp.append(next_features)
                        v_t_inp.append(features)
                        d_ts.append(d_t)
                        log_pis.append(torch.log(policy.probs[action]))
                        entropies.append(policy.entropy().mean())
                    
        
            if finished:
                break
            if is_learn and  switching_agent.trainable and len(td_errors):
                td_errors = torch.stack(td_errors)
                
                switching_agent.update_policy(1, td_errors)
                if torch.is_tensor(list(switching_agent.network.parameters())[0].grad):
                    if not torch.any(list(switching_agent.network.parameters())[0].grad > 0.):
                        print('critic zero grad ', file=sys.stderr)
                    if not torch.all(list(switching_agent.network.parameters())[-1].grad < 1e3):
                        print('critic grad > 1e3', file=sys.stderr)
                        

            
            if is_learn and acting_agents[1].trainable and len(costs_for_delta):             
               
                with torch.no_grad():
                    v_tplus1 = switching_agent.target_network(v_tplus1_inp)
                    v_t = switching_agent.network(v_t_inp)
            
                deltas = torch.as_tensor(costs_for_delta) + v_tplus1 - v_t
                if not deltas.any():
                    print('Deltas', file=sys.stderr)                 
                
                log_pis = torch.stack(log_pis)
                entropies = torch.stack(entropies)
                d_ts = torch.as_tensor(d_ts)
                acting_agents[1].update_policy(d_ts, deltas, log_pis, entropies)
                
                if torch.is_tensor(list(acting_agents[1].network.parameters())[0].grad):
                    if not torch.any(list(acting_agents[1].network.parameters())[0].grad > 0.):
                        print('actor zero grad ', file=sys.stderr)
                    if not torch.all(list(acting_agents[1].network.parameters())[-1].grad < 1e3):
                        print('actor grad > 1e3', file=sys.stderr)
                        print(deltas, log_pis, entropies, current_state, d_t, action, policy.probs)



            trajectory_cost += c_tplus1            
        if ret_trajectory:
            trajectories.append(trajectory)
            
        total_costs.append(trajectory_cost)
   
    if ret_trajectory:
        return trajectories
                    
    
    return np.mean(total_costs), np.mean(total_machine_picked)



def learn_off_policy(switching_agent: Agent, acting_agents, trip_id, n_try=1):
    """
    Learn  overall policy off-policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The actings agents (always contains 2)

    trip_id: int 
        The  trip id induced by the behavior policy. 
        
    
    Returns
    -------
    np.mean(machine_picked) : float
        Average times machine was picked
    """

    
  
    machine_picked = []
    for i in range(n_try):
        
        critic_emphatic_weightings = []
        td_errors = []
        actor_emphatic_weightings = []
        deltas = []
        log_pis = []
        entropies = []
        costs_for_delta = []
        v_tplus1_inp = []
        v_t_inp = []
        print(f"Getting source and destination for trip_id {trip_id}")
        start_node_id = TRIPS[trip_id][0]
        finish_node_id = TRIPS[trip_id][1]
        print("Reseting env")

        # TODO: fix this nicely b = idx of element on batch now batch_size = 1
        b = 0
        ENV.reset(start_node_id,finish_node_id)
        finished = False
        while not finished : 
            current_state = ENV.current_state()
            action, next_state, cost,  finished = ENV.next_human_step(trip_id)
            d_t = switching_agent.take_action(current_state, train=True)
            machine_picked.append(d_t)
            option = acting_agents[d_t]          
                        
            c_tplus1 = cost + option.control_cost
            
            if switching_agent.trainable:
                next_features = Environment.state2features(next_state, switching_agent.n_state_features) 
                with torch.no_grad():
                    d_tplus1 = switching_agent.take_action(next_state, train=True, use_target=True)
                    if switching_agent.network.needs_agent_feature :                        
                        next_features = [*next_features, *Environment.agent_feature2net_input(d_tplus1)]
                    v_tplus1 = switching_agent.target_network(next_features)

                features = Environment.state2features(current_state, switching_agent.n_state_features)
                if switching_agent.network.needs_agent_feature :
                    features = [*features, *Environment.agent_feature2net_input(d_t)]
                v_t = switching_agent.network(features)
                
                td_error = c_tplus1 + v_tplus1 - v_t
                # behavior policy
                mu_t = acting_agents[0].get_policy(current_state, action, next_state)
                # target policy
                policy = acting_agents[1].take_action(current_state)[1]
                with torch.no_grad():

                    var_rho_prev = switching_agent.var_rho[b]
                    i_s = 1

                    switching_agent.F_t[b] = i_s + var_rho_prev * switching_agent.F_t[b]

                    machine_pi_t = policy.probs[action].item()

                    rho = machine_pi_t / mu_t

                    var_pi_t = machine_pi_t if d_t else mu_t
                    switching_agent.var_rho[b] = var_pi_t / mu_t

                emphatic_weighting  = switching_agent.var_rho[b] * switching_agent.F_t[b]                    
                
                if not td_error:
                    print("TD error")
                    print(c_tplus1, v_tplus1, v_t)  
                    print(current_state, action, next_state)                    

                if not emphatic_weighting:
                    print('critic emphatic', policy.probs,mu_t,  switching_agent.F_t[b], var_rho_prev)
                                    
                critic_emphatic_weightings.append(emphatic_weighting)
                td_errors.append(td_error)
                
            if acting_agents[1].trainable:      
                acting_agents[1].M_t[b] = d_t + var_rho_prev*acting_agents[1].M_t[b]
                emphatic_weighting = rho * acting_agents[1].M_t[b]
                if not emphatic_weighting:
                    print('actor emphatic ',rho, var_rho_prev, acting_agents[1].M_t[b] )
                
                actor_emphatic_weightings.append(emphatic_weighting)
                costs_for_delta.append(cost)
                v_tplus1_inp.append(next_features)
                v_t_inp.append(features)

                log_pis.append(torch.log(policy.probs[torch.as_tensor(action)]))
                entropies.append(policy.entropy().mean())

            if switching_agent.trainable and len(td_errors):
                critic_emphatic_weightings = torch.as_tensor(critic_emphatic_weightings)                
                td_errors = torch.stack(td_errors)
                
                switching_agent.update_policy(critic_emphatic_weightings, td_errors)

                if torch.is_tensor(list(switching_agent.network.parameters())[0].grad):
                    if not torch.any(list(switching_agent.network.parameters())[0].grad > 0.):
                        print('critic zero grad ', file=sys.stderr)
                        pass
                    if not torch.all(list(switching_agent.network.parameters())[-1].grad < 1e3):
                        print('critic grad > 1e3', file=sys.stderr)
            if acting_agents[1].trainable and len(actor_emphatic_weightings):
            
                with torch.no_grad():
                    v_tplus1 = switching_agent.target_network(v_tplus1_inp)
                    v_t = switching_agent.network(v_t_inp)
                    
                deltas = torch.as_tensor(costs_for_delta) + v_tplus1 - v_t
                                        
                if not deltas.any():
                    print('Deltas', file=sys.stderr) 
                    
                actor_emphatic_weightings = torch.as_tensor(actor_emphatic_weightings)
                log_pis = torch.stack(log_pis)
                entropies = torch.stack(entropies)

                acting_agents[1].update_policy(actor_emphatic_weightings, deltas, log_pis, entropies, use_entropy=False)
                
                if torch.is_tensor(list(acting_agents[1].network.parameters())[0].grad):
                    if not torch.any(list(acting_agents[1].network.parameters())[0].grad > 0.):
                        print('actor zero grad', file=sys.stderr)
                        print(actor_emphatic_weightings,log_pis,deltas , file=sys.stderr)
                        
                    if not torch.all(list(acting_agents[1].network.parameters())[-1].grad < 1e3):
                        print('actor grad > 1e3', file=sys.stderr)
                        print(rho, var_rho_prev, acting_agents[1].M_t[0],actor_emphatic_weightings, deltas, log_pis, entropies, current_state, d_t, action, policy.probs)

    return np.mean(machine_picked)
          
 


    
