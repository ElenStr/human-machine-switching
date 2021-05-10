"""
Environment types in the paper for sensor-based state spaces.
"""

from environments.episodic_mdp import EpisodicMDP
from environments.env import Environment, GridWorld
import numpy as np
from copy import copy
import math
import datetime

# default width and height
WIDTH, HEIGHT = 3, 10
# default episode length
EP_L = HEIGHT


# Keep and adjust only state2feature for networks' input but can be included in some utils.py 
class FeatureStateHandler:
    """
    Convert a feature vector to a state number and vice versa.
    For example, if the cell types are ['road', 'grass', 'car'], and traffic levels
    are ['no-car', 'light', 'heavy'], then s = ('no-car', 'road', 'grass', 'road', 'car')
    <==> s = (00102) in base 3 = 11
    """

    def __init__(self, types: list, traffic_levels: list):
        """
        Parameters
        ----------
        types : list of str
            All the cell types
        traffic_levels : list of str
            All traffic levels
        """

        self.types = types
        # types= ['road', 'car', ...]
        if 'wall' not in self.types:
            self.types.append('wall')

        self.traffic_levels = traffic_levels

        self.type_value = {}
        for i, t in enumerate(self.types):
            self.type_value[t] = i

        self.traffic_value = {}
        for i, t in enumerate(self.traffic_levels):
            self.traffic_value[t] = i

        self.base = len(self.types)
       
        self.n_types_feature = 4
      
        self.n_state = np.power(self.base, self.n_types_feature) * len(self.traffic_levels)

    def feature2state(self, feature_vec: list):
        """
        Parameters
        ----------
        feature_vec : list of str
            An input feature vector. The dimension should be
            equal to `self.n_types_feature + 1`

        Returns
        -------
        state_num : int
            The state number corresponding to the input feature vector
        """
        
        assert len(feature_vec) == self.n_types_feature + 1, 'input dimension not equal to the feature dimension'

        traffic_value = self.traffic_value[feature_vec[0]]
        types_feature_vec = feature_vec[1:]
        types_values = [self.type_value[g] for g in types_feature_vec]
        state_num = 0
        for i, value in enumerate(types_values):
            state_num += value * (self.base ** (self.n_types_feature - i - 1))

        state_num = self.base ** self.n_types_feature * traffic_value + state_num
        return state_num

    def state2feature(self, state: int):
        """
        Parameters
        ----------
        state : int
            An state number. It should satisfy the condition
            `0 <= state < self.n_state`

        Returns
        -------
        feature_vec : list
            The feature vector corresponding to the input state number
        """
        assert 0 <= state < self.n_state, 'invalid state number'
       

        traffic_value = state // (self.base ** self.n_types_feature)
        state -= traffic_value * (self.base ** self.n_types_feature)

        type_numbers = list(map(int, np.base_repr(state, self.base)))
        for j in range(self.n_types_feature - len(type_numbers)):
            type_numbers.insert(0, 0)

        traffic_level = [self.traffic_levels[traffic_value]]
        feature_vec = traffic_level + [self.types[i] for i in type_numbers]
        return feature_vec

# Not needed see state function in gridworld class
def make_state_extractor(env: Environment):
    types = copy(env.cell_types)
    env_traffic_levels = copy(env.traffic_levels)
    feat2state = FeatureStateHandler(types=types, traffic_levels=env_traffic_levels).feature2state

    def sensor_state_extractor(grid_env: GridWorld, coordinate: tuple):
        cell_types = grid_env.cell_types
        traffic_levels = grid_env.traffic_levels

        x, y = coordinate
        f_s = [traffic_levels[y + 1], cell_types[x, y]]
        
        f_s.extend([cell_types.get((x + i - 1, y + 1), 'wall') for i in range(3)])
        return feat2state(f_s)

    return sensor_state_extractor

# Not needed anymore 
def make_sensor_based_mdp(env: Environment, ep_l: int = EP_L, verbose=True):
    """
    Generate an episodic MDP with sensor-based state space.

    Parameters
    ----------
    env : Environment
        Specifies the environment type defined in the paper
    ep_l : int
        Episode length
    verbose: bool
        If it prints the log

    Returns
    -------
    ep_env : EpisodicMDP
        The sensor-based episodic MDP based on the environment
    """

    def _find_pos(f):
        # positions: 0=left, 1=middle, 2=right
        if f[1] == 'wall':
            return 0
        if f[3] == 'wall':
            return 2
        return 1

    def _calc_prob(f_s, f_sn, t_lvl_n, a):
        pos = _find_pos(f_s)
        pos_n = _find_pos(f_sn)

        # raise exception when the action goes to wall
        if a != 1 and pos == a:
            raise Exception('The action goes to wall')

        check_correct_pos = pos + a - 1 == pos_n
        if not check_correct_pos or f_sn[0] != f_s[a + 1]:
            return 0
        
        middle_cell = 3 - pos_n
        middle_prob = type_probs[t_lvl_n][f_sn[middle_cell]]
        # calculate the probability
        if pos_n == 0:
            return 1 * type_probs[t_lvl_n][f_sn[2]] * middle_prob
        if pos_n == 1:
            return type_probs[t_lvl_n][f_sn[1]] * middle_prob * type_probs[t_lvl_n][f_sn[3]]

        return middle_prob * type_probs[t_lvl_n][f_sn[2]] * 1

    # actions = [left, straight, right]
    n_action = 3

    type_probs = copy(env.type_probs)
    traffic_probs = copy(env.traffic_probs)
    type_costs = copy(env.type_costs)

    # add 'wall' type
    for traffic in env.traffic_levels:
        type_probs[traffic]['wall'] = 0
    type_costs['wall'] = max(type_costs.values()) + 1

    f_s_handler = FeatureStateHandler(types=copy(env.cell_types), traffic_levels=copy(env.traffic_levels))

    # number of states
    n_state = f_s_handler.n_state

    # true costs and transitions
    true_costs = np.zeros(shape=(n_state, n_action))
    true_transitions = np.zeros(shape=(n_state, n_action, n_state))
    count = 0
    for state in range(n_state):
        for action in range(n_action):
            feat_vec = f_s_handler.state2feature(state)

            type_feat_vec = feat_vec[1:]
            traffic_lvl = feat_vec[0]
            
            true_costs[state][action] = type_costs[type_feat_vec[0]]

            # handle wall
            real_action = action
            if _find_pos(type_feat_vec) == action:
                real_action = 1

            for nxt_state in range(n_state):
                nxt_feat_vec = f_s_handler.state2feature(nxt_state)
                nxt_type_feat_vec = nxt_feat_vec[1:]
                nxt_traffic_lvl = nxt_feat_vec[0]
                
                true_transitions[state][action][nxt_state] = (
                        traffic_probs[traffic_lvl][nxt_traffic_lvl]
                        * _calc_prob(type_feat_vec, nxt_type_feat_vec, nxt_traffic_lvl, real_action)
                )

            # Normalize TODO do we need it?
            true_transitions[state][action] = true_transitions[state][action] / sum(true_transitions[state][action])
        count += 1
        if verbose and count % 200 == 0:
            print(f'{datetime.datetime.now()}\t{int(count/n_state * 100)}%')

    ep_env = EpisodicMDP(n_state, n_action, ep_l, true_costs, true_transitions)
    return ep_env