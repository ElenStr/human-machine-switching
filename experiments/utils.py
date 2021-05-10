import random

from agent.agents import Agent
from agent.switching_agents import SwitchingAgent
from environments.env import Environment, GridWorld
from environments.make_envs import WIDTH, HEIGHT, make_state_extractor
from plot.plot_path import HUMAN_COLOR, MACHINE_COLOR, PlotPath

env = Environment()
feat_ext = make_state_extractor(env)


def get_trajectory_cost(agents, init_traffic, n_try=1):
    trajectory = env.generate_grid_world(width=WIDTH, height=HEIGHT, init_traffic_level=init_traffic)
    costs = []
    for agent in agents:
        costs.append(evaluate(agent=agent, env=trajectory, feat_ext=feat_ext, n_try=n_try))

    return costs


def evaluate(agent: Agent, env: GridWorld, feat_ext, n_try=1, plt_path=None):
    """
    Learn or evaluate a switching policy in a grid environment.

    Parameters
    ----------
    is_learn : bool
        Indicates if we are training or evaluating. If `is_learn == True`,
        then `n_try = 1`, and we will update the observations for the learning agent

    feat_ext: function
        Extracts the feature vector of each state

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """
    total_costs = 0
    count = 0
    for i in range(n_try):
        time = 0
        count += 1
        finished = False
        env.reset()
        s_t= env.current_coord
        while not finished:
            feat_state = feat_ext(env, s_t)
            a_t = agent.take_action(feat_state, time)
            s_tplus1, cost, finished = env.next_cell(a_t)

            if not finished:
                total_costs += cost

            # update observations
            if not finished:
                agent.update_obs(feat_ext(env, s_t), a_t, feat_ext(env, s_tplus1))

            if plt_path is not None and not finished:
                clr = MACHINE_COLOR
                plt_path.add_line(s_t, s_tplus1, clr)

            s_t = s_tplus1
            time += 1

    return total_costs / count


def evaluate_switching(switching_agent: SwitchingAgent, agents_cost: list, switching_cost: float,
                       trajectory: GridWorld, agents: list, feat_ext, n_try=1, plt_path=None):
    """
    Evaluate (learn) a switching policy in a trajectory.

    Parameters
    ----------
    feat_ext: function
        Extracts the feature vector of each state

    Returns
    -------
    total_cost : int
        Average total cost of the trajectory
    """

    total_costs = 0
    for i in range(n_try):
        time = 0
        finished = False
        trajectory.reset()
        d_tminus1 = 0
        s_t = trajectory.current_coord
        while not finished:
            d_t = switching_agent.take_action(feat_ext(trajectory, s_t), d_tminus1, time)
            a_t = agents[d_t].take_action(feat_ext(trajectory, s_t), time)

            s_tplus1, env_cost, finished = trajectory.next_cell(a_t)

            is_switch = int(d_tminus1 != d_t) * int(time > 0)
            cost = env_cost + switching_cost * is_switch + agents_cost[d_t]

            if not finished:
                total_costs += cost

            # update observations
            if not finished:
                switching_agent.update_obs(d_tminus1, feat_ext(trajectory, s_t), d_t, a_t, env_cost, feat_ext(trajectory, s_tplus1),
                                           finished)

            if plt_path is not None and not finished:
                clr = MACHINE_COLOR if d_t == 0 else HUMAN_COLOR
                plt_path.add_line(s_t, s_tplus1, clr)
            s_t = s_tplus1
            d_tminus1 = d_t
            time += 1

    return total_costs / n_try


def test_agent_policy(agent: Agent, path: str):
    init_traffic = random.choice(['no-car', 'light', 'heavy'])
    grid_env = Environment().generate_grid_world(width=WIDTH, height=HEIGHT, init_traffic_level=init_traffic)
    plt_path = PlotPath(grid_env, n_try=1)
    feat_ext = make_state_extractor(Environment())
    cost = evaluate(agent=agent, env=grid_env, feat_ext=feat_ext,
                    n_try=1, plt_path=plt_path)
    plt_path.plot(path)
    return cost


def test_switching_agent(switching_agent: SwitchingAgent, agents: list, n_try: int, path: str, agents_costs: list,
                         switching_cost: float, init_traffic: str = None):
    if init_traffic is None:
        init_traffic = random.choice(['no-car', 'light', 'heavy'])
    grid_env = Environment().generate_grid_world(width=WIDTH, height=HEIGHT, init_traffic_level=init_traffic)
    plt_path = PlotPath(grid_env, n_try)
    feat_ext = make_state_extractor(Environment())
    cost = evaluate_switching(switching_agent, agents_costs, switching_cost, grid_env, agents, feat_ext,
                              n_try, plt_path)
    plt_path.plot(path)
    return cost
