"""
Implementation of the switching agent in the paper
"""
import numpy as np
import copy
from environments.episodic_mdp import EpisodicMDP, TEST_MDP
from numba import njit
from numba.typed import List
from agent.agents import UCRL2, Agent, ConfidenceSet, FixedAgent


class SwitchingAgent(Agent):
    """
    An agent that decides the control switch at each time, given the current state and
    the previous control switch (i.e., $\pi_t(d_t|s_t, d_{t-1})$ in the paper).

    Child agent will implement `update_policy` function.
    """

    def __init__(self, env: EpisodicMDP, agents: list, switching_cost: float, agents_cost: list):
        """
        The initial controller agent is agent0

        Parameters
        ----------
        env : EpisodicMDP
            The environment object
        agents : list of Agent
            The agent' action policies (i.e., human/machine action policies)
        switching_cost: float
            The costs of switching between two agent
        agents_cost: list of float
            The costs of choosing each agent as the controller
        """
        super().__init__()
        self.agents = copy.deepcopy(agents)
        self.env = env
        self.switching_cost = switching_cost
        self.agents_cost = agents_cost

        # initialize the current agent as agent 0
        self.cur_agent = 0

        # initialize the switching policy [time_step][prev_agent][state]: agent 0
        self.policy = np.zeros(shape=(self.env.ep_l, len(self.agents), self.env.n_state), dtype=int)

    def take_action(self, state, prev_agent, time_step):
        """chooses the controller of the current state"""
        self.cur_agent = int(self.policy[time_step][prev_agent][state])
        return self.cur_agent

    def update_obs(self, d_tminus1, s_t, d_t, action, env_cost, s_tplus1, finished):
        """ update agent histories """
        pass


class UCRL2MC(SwitchingAgent):
    def __init__(self, env, agents, switching_cost, agents_cost, delta, unknown_agents: list,
                 env_prior: ConfidenceSet = None, time_changing=None, total_episodes=10000):
        super().__init__(env, agents, switching_cost, agents_cost)

        # index of unknown agent
        self.unknown_agents = unknown_agents

        self.delta = delta
        self.episode_num = 0

        self.total_episodes = total_episodes

        self.time_changing = time_changing
        if time_changing is None:
            self.time_changing = []

        # history of unknown agent [agent][state][action]
        self.agents_prior = []
        for i in range(len(self.agents)):
            prior = None
            shape = (self.env.n_state, self.env.n_action)
            if i not in self.unknown_agents:
                prior = self.agents[i].policy
            self.agents_prior.append(ConfidenceSet(shape=shape, delta=self.delta, true_policy=prior))

        # history of the environment transitions [state][action][next_state]
        if env_prior is None:
            env_prior = ConfidenceSet(shape=(self.env.n_state, self.env.n_action, self.env.n_state), delta=self.delta)
        self.env_prior = env_prior

        self.beta_ags = None
        self.p_ags = None

        self.beta_env = None
        self.p_env = None

        self.value_function = np.zeros(shape=(self.env.ep_l + 1, len(self.agents), self.env.n_state))

        for t in range(self.env.ep_l):
            for s in range(self.env.n_state):
                for d in range(len(self.agents)):
                    self.value_function[t][d][s] = np.min(self.env.costs[s])

    def update_obs(self, d_tminus1, s_t, d_t, action, env_cost, s_tplus1, finished):
        self.agents_prior[d_t].update((s_t, action))
        self.env_prior.update((s_t, action, s_tplus1))

    def update_policy(self, ep_num):
        scale_env = max(0.1 - (ep_num // (self.total_episodes / 10)) * 0.01, 0.01)
        scale_agent = max(1 - (ep_num // (self.total_episodes / 3000)) * 0.1, 0.1)

        self.episode_num = ep_num
        t_k = ep_num * self.env.ep_l + 1

        self.beta_ags = [agent_prior.get_beta(t_k) * scale_agent for agent_prior in self.agents_prior]
        self.p_ags = [agent_prior.get_empirical() for agent_prior in self.agents_prior]

        t_k_env = np.sum(self.env_prior.history) + 1
        self.beta_env = self.env_prior.get_beta(t_k_env) * scale_env
        self.p_env = self.env_prior.get_empirical()

    def take_action(self, state, prev_agent, time_step):
        env_c, n_s, n_ac = self.env.costs[state], self.env.n_state, self.env.n_action
        s_c = self.switching_cost
        ags_c = List()
        [ags_c.append(float(c)) for c in self.agents_cost]

        n_ags = len(self.agents)

        p_env = self.p_env[state]
        beta_env = self.beta_env[state]

        p_ags = List()
        beta_ags = List()

        for ag in range(len(self.agents)):
            if ag in self.time_changing:
                p_ags.append(self.p_ags[ag][time_step][state])
            else:
                p_ags.append(self.p_ags[ag][state])

        [beta_ags.append(beta_ag[state]) for beta_ag in self.beta_ags]

        nxt_val_func = self.value_function[time_step + 1]

        q_value = self.calculate_minimization(nxt_val_func, p_env, beta_env, p_ags, beta_ags, env_c, n_s, n_ac,
                                              s_c, ags_c, n_ags, time_step)

        best_agents = np.where(q_value[prev_agent] == np.min(q_value[prev_agent]))[0]
        best_agent = np.random.choice(best_agents)
        self.policy[time_step][prev_agent][state] = best_agent

        for d in range(len(self.agents)):
            self.value_function[time_step][d][state] = max(self.value_function[time_step][d][state], min(q_value[d]))

        return int(self.policy[time_step][prev_agent][state])

    @staticmethod
    @njit
    def calculate_minimization(nxt_value_func, p_env, beta_env, p_ags, beta_ags, env_c, n_s, n_ac, s_c, ags_c, n_ags, t):

        q_val = np.zeros(shape=(n_ags, n_ags))
        zero = 1e-5
        for d in range(n_ags):
            # minimize p_env
            p_env_opt = np.copy(p_env)
            for action in range(n_ac):
                if beta_env[action] > zero:
                    sorted_states = np.argsort(nxt_value_func[d])
                    if p_env_opt[action][sorted_states[0]] + 0.5 * beta_env[action] > 1:
                        p_env_opt[action] = np.zeros(n_s)
                        p_env_opt[action][sorted_states[0]] = 1
                    else:
                        p_env_opt[action][sorted_states[0]] += 0.5 * beta_env[action]

                    j = n_s - 1
                    while p_env_opt[action].sum() > 1:
                        p_env_opt[action][sorted_states[j]] = (
                            max(1 - p_env_opt[action].sum() + p_env_opt[action][sorted_states[j]], 0)
                        )
                        j -= 1

            action_vals = np.array([env_c[a] + np.dot(p_env_opt[a], nxt_value_func[d]) for a in range(n_ac)])
            p_ags_opt = np.copy(p_ags[d])
            if beta_ags[d] > zero:
                sorted_actions = np.argsort(action_vals)

                # optimize p_ags_opt
                if p_ags_opt[sorted_actions[0]] + 0.5 * beta_ags[d] > 1:
                    p_ags_opt = np.zeros(n_ac)
                    p_ags_opt[sorted_actions[0]] = 1
                else:
                    p_ags_opt[sorted_actions[0]] += 0.5 * beta_ags[d]

                j = n_ac - 1
                while p_ags_opt.sum() > 1:
                    p_ags_opt[sorted_actions[j]] = max(1 - p_ags_opt.sum() + p_ags_opt[sorted_actions[j]], 0)
                    j -= 1

            value = np.dot(action_vals, p_ags_opt)
            for dm in range(n_ags):
                meta_cost = ags_c[d] + int(dm != d) * s_c * int(t != 0)
                q_val[dm][d] = meta_cost + value

        return q_val


class OptimalSwitching(SwitchingAgent):

    def __init__(self, env, agents, switching_cost, agents_cost, time_changing=None):
        super().__init__(env, agents, switching_cost, agents_cost)
        self.time_changing = time_changing
        if time_changing is None:
            self.time_changing = []

        env_c, n_s, n_ac, ep_l = self.env.costs, self.env.n_state, self.env.n_action, self.env.ep_l
        s_c = self.switching_cost

        ags_c = List()
        [ags_c.append(float(c)) for c in self.agents_cost]

        p_env = self.env.trans_probs

        p_ags = List()
        for i, agent in enumerate(self.agents):
            policy = agent.policy
            if i not in self.time_changing:
                policy = np.repeat(policy[np.newaxis, :, :], ep_l, axis=0)
            p_ags.append(policy)

        self.policy = self.val_itr(p_ags, p_env, env_c, n_s, n_ac, ep_l, s_c, ags_c)

    @staticmethod
    @njit
    def val_itr(p_ags, p_env, env_c, n_s, n_ac, ep_l, s_c, ags_c):

        n_ags = len(ags_c)
        # q_val[time][prev_agent][state][curr_agent]
        q_val = np.zeros(shape=(ep_l, n_ags, n_s, n_ags))
        # q_min[time][prev_agent][state]
        q_min = np.zeros(shape=(ep_l + 1, n_ags, n_s))

        # policy[time][prev_agent][state]
        policy = np.zeros(shape=(ep_l, n_ags, n_s))

        for i in range(ep_l):
            t = ep_l - i - 1
            for s in range(n_s):
                for d in range(n_ags):
                    action_vals = np.array([env_c[s][a] + np.dot(p_env[s][a], q_min[t + 1][d]) for a in range(n_ac)])

                    value = np.dot(action_vals, p_ags[d][t][s])

                    for dm in range(n_ags):
                        meta_cost = ags_c[d] + int(dm != d) * s_c * int(t != 0)
                        q_val[t][dm][s][d] = meta_cost + value

                for dm in range(n_ags):
                    # update policy
                    best_agents = np.where(q_val[t][dm][s] == np.min(q_val[t][dm][s]))[0]
                    best_agent = np.random.choice(best_agents)
                    policy[t][dm][s] = best_agent
                    q_min[t][dm][s] = q_val[t][dm][s][best_agent]

        return policy


class UCRL2Switching(SwitchingAgent):
    def __init__(self, env, agents, switching_cost, agents_cost, delta, total_episodes=10000):

        super().__init__(env, agents, switching_cost, agents_cost)

        self.env = env
        self.episode_num = 0
        self.delta = delta
        self.total_episodes = total_episodes

        self.env_prior = ConfidenceSet(shape=(self.env.n_state, len(self.agents), self.env.n_state), delta=self.delta)

        self.value_function = np.zeros(shape=(self.env.ep_l + 1, len(self.agents), self.env.n_state))

        for t in range(self.env.ep_l):
            for s in range(self.env.n_state):
                for d in range(len(self.agents)):
                    self.value_function[t][d][s] = np.min(self.env.costs[s])

        self.beta_env = None
        self.p_env = None
        self.t_k = 0

    def update_obs(self, d_tminus1, s_t, d_t, action, env_cost, s_tplus1, finished):
        self.env_prior.update((s_t, d_t, s_tplus1))

    def update_policy(self, ep_num):
        scale_env = max(0.1 - (ep_num // (self.total_episodes / 10)) * 0.01, 0.01)

        self.episode_num = ep_num
        t_k = ep_num * self.env.ep_l + 1

        self.beta_env = self.env_prior.get_beta(t_k) * scale_env
        self.p_env = self.env_prior.get_empirical()

    @staticmethod
    @njit
    def calculate_minimization(nxt_value_func, p_env, beta_env, costs, s_c, n_s, n_ags, t):
        q_val = np.zeros(shape=(n_ags, n_ags))
        zero = 1e-5
        for d in range(n_ags):
            # minimize p_env
            p_env_opt = np.copy(p_env[d])
            if beta_env[d] > zero:
                sorted_states = np.argsort(nxt_value_func[d])
                if p_env_opt[sorted_states[0]] + 0.5 * beta_env[d] > 1:
                    p_env_opt = np.zeros(n_s)
                    p_env_opt[sorted_states[0]] = 1
                else:
                    p_env_opt[sorted_states[0]] += 0.5 * beta_env[d]

                j = n_s - 1
                while p_env_opt.sum() > 1:
                    p_env_opt[sorted_states[j]] = max(1 - p_env_opt.sum() + p_env_opt[sorted_states[j]], 0)
                    j -= 1

            value = costs[d] + np.dot(p_env_opt, nxt_value_func[d])
            for dm in range(n_ags):
                meta_cost = int(dm != d) * s_c * int(t != 0)
                q_val[dm][d] = meta_cost + value

        return q_val

    def take_action(self, state, prev_agent, time_step):
        costs = List()
        [costs.append(float(self.env.costs[state].mean() + ag_c)) for ag_c in self.agents_cost]
        n_s, n_ags, ep_l = self.env.n_state, len(self.agents), self.env.ep_l

        s_c = self.switching_cost

        p_env = self.p_env[state]
        beta_env = self.beta_env[state]

        next_value_function = self.value_function[time_step + 1]

        q_value = self.calculate_minimization(next_value_function, p_env, beta_env, costs, s_c, n_s, n_ags, time_step)

        best_agents = np.where(q_value[prev_agent] == np.min(q_value[prev_agent]))[0]
        best_agent = np.random.choice(best_agents)
        self.policy[time_step][prev_agent][state] = best_agent

        for d in range(len(self.agents)):
            self.value_function[time_step][d][state] = max(self.value_function[time_step][d][state], min(q_value[d]))

        return int(self.policy[time_step][prev_agent][state])