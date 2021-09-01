from collections import defaultdict

import torch
import sys
import datetime

from agent.agents import Agent
from environments.taxi_env import MapEnv, FeatureHandler
from environments.utils_env import *


def taxi_learn_evaluate(switching_agent: Agent, acting_agents: list, env: MapEnv, initial_state: tuple, is_learn: bool,
                        state2feature, max_ep_l: int, n_try: int = 1):

    """
    Learn (on policy) or evaluate overall policy in a grid environment.

    Parameters
    ----------
    switching_agent:  Agent
        The switching agent

    acting_agents:  list of Agent
        The acting agents

        The environment

    initial_state: tuple
        The initial state

    max_ep_l: int
        Maximum number of steps if the episode does not reach a termination state

    is_learn: bool
        Indicates if we are training or evaluating. If `is_learn == True`,
        then `n_try = 1`, and we will update the policy for the agents + switching policy

    online_ev: bool
        True in online evaluation

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """
    total_costs = []
    total_machine_picked = []

    for i in range(n_try):
        timestep = 0
        trajectory_cost = 0
        finished = False

        env.reset(initial_state)
        while timestep < max_ep_l and not finished:
            timestep += 1
            current_state = env.current_state()

            d_t = switching_agent.take_action(current_state)

            option = acting_agents[d_t]
            total_machine_picked.append(d_t)
            action, policy = option.take_action(current_state)

            next_state, cost, finished = env.step(action)
            c_tplus1 = cost + option.control_cost

            if is_learn:
                next_features = state2feature(next_state)
                with torch.no_grad():
                    d_tplus1 = switching_agent.take_action(next_state)
                    if switching_agent.network.needs_agent_feature:
                        next_features = [*next_features, *get_agent_feature(d_tplus1)]
                    v_tplus1 = switching_agent.target_network(next_features) if not finished else 0

                current_features = state2feature(current_state)
                if switching_agent.network.needs_agent_feature:
                    current_features = [*current_features, *get_agent_feature(d_t)]
                v_t = switching_agent.network(current_features)
                td_error = c_tplus1 + v_tplus1 - v_t
                if switching_agent.trainable:
                    if td_error == 0.:
                        print('TD Error')
                        print(c_tplus1, v_tplus1, v_t)

                    switching_agent.update_policy(1, td_error)
                    #
                    if torch.is_tensor(list(switching_agent.network.parameters())[0].grad):
                        # TODO: handle critic zero grad
                        if not torch.any(list(switching_agent.network.parameters())[0].grad > 0.):
                            print('critic zero grad ', file=sys.stderr)
                        if not torch.all(list(switching_agent.network.parameters())[-1].grad < 1e3):
                            print('critic grad > 1e3', file=sys.stderr)

                if option.trainable:
                    with torch.no_grad():
                        v_t = switching_agent.target_network(current_features)
                    delta = cost + v_tplus1 - v_t
                    log_pi = torch.log(policy.probs[action])
                    entropy = policy.entropy().mean()
                    acting_agents[1].update_policy(d_t, delta, log_pi, entropy)

                    if torch.is_tensor(list(acting_agents[1].network.parameters())[0].grad):
                        if not torch.any(list(acting_agents[1].network.parameters())[0].grad > 0.):
                            print('actor zero grad ', file=sys.stderr)
                        if not torch.all(list(acting_agents[1].network.parameters())[-1].grad < 1e3):
                            print('actor grad > 1e3', file=sys.stderr)
                            print(delta, log_pi, entropy, current_state, d_t, action, policy.probs)

            trajectory_cost += c_tplus1

        total_costs.append(trajectory_cost)

    return np.mean(total_costs), np.mean(total_machine_picked)


def taxi_train_online(algos, online_set, eval_set, env: MapEnv, feature_handler: FeatureHandler, eval_freq: int,
                      max_ep_l: int, verbose: bool = True, eval_tries=1):
    """
    Train the switching and machine policy for different configurations
    of machine and switching agents.

    Parameters
    ---------
    algos: dict of 'algorithm_name' : (switching_agent, [human, machine])
        The switching agents and the acting agents to be trained

    online_set: list
        list of initial states for the MapEnv environment

    eval_set:
        Evaluation set of environments to keep track on training progress

    eval_freq: int
        Agents' evaluation frequency

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
    algos_costs = defaultdict(lambda: [])
    machine_picked_ratios = defaultdict(lambda: [])
    machine_picked_ratios_tr = defaultdict(lambda: [])
    for ep, init_state in enumerate(online_set):
        print(ep)
        ep += 1
        for algo, agents in algos.items():
            switching_agent, acting_agents = agents
            machine = acting_agents[1]

            _, machine_picked_tr = taxi_learn_evaluate(switching_agent, acting_agents, env, init_state,
                                                       max_ep_l=max_ep_l , state2feature=feature_handler.state2feature,
                                                       is_learn=True)
            machine_picked_ratios_tr[algo].append(machine_picked_tr)
            # print log
            if verbose and ep % eval_freq == 0 and (ep // eval_freq > 0):
                eval_cost = 0
                machine_picked = 0
                for state in eval_set:
                    x, y = taxi_learn_evaluate(switching_agent, acting_agents, env, state,
                                                                   max_ep_l=max_ep_l, n_try=eval_tries,
                                                                   is_learn=False, state2feature=feature_handler.state2feature)
                    eval_cost += x
                    machine_picked += y
                machine_picked /= len(eval_set)
                eval_cost /= len(eval_set)
                print(f'{datetime.datetime.now()}, On-policy, Episode {ep}, {algo}  evaluation cost: {eval_cost}')
                algos_costs[algo].append(eval_cost)
                if 'switch' in algo or 'fxd' in algo:
                    print(machine_picked)
                    print(np.mean(machine_picked_ratios_tr[algo]))
                    machine_picked_ratios[algo].append(machine_picked)

                    # save agent
            # if save_agent and (ep % save_freq == 0) and (ep // save_freq > 0):
            #     save_agent_cost(algo, machine, switching_agent, algos_costs[algo], machine_picked_ratios[algo],
            #                     'on')
            algos[algo] = (switching_agent, acting_agents)

    return algos, algos_costs


def get_agent_feature(value):
    f = [0., 0.]
    f_pos = 1 - value
    f[f_pos] = 1.

    return f
