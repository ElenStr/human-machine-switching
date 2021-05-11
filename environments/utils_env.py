"""
Environment help functions.
"""

import numpy as np
from copy import copy
import math
from environments.env import CELL_TYPES, TRAFFIC_LEVELS, Environment
# default width and height
WIDTH, HEIGHT = 3, 10
# default max episode length
EP_L = HEIGHT


def state2features(state, env: Environment):
    """
    Parameters
    ----------
    state: list of strings
        Current cell type and traffic factor and cell types of next rows 
    env: Environment
        Current environment

    Returns
    -------
    features: array of int
        The state feature vector
    """
    cell_t = np.array(CELL_TYPES.append('wall'))
    traffic_l = np.array(TRAFFIC_LEVELS)
    features = [np.argwhere(cell_t == state[0])[0][0]] 
    for i in range(1,len(state)):
        if i-1 % (WIDTH+1) == 0:
            features.append(np.argwhere(traffic_l == state[i])[0][0])
        else:
            features.append(np.argwhere(cell_t == state[i])[0][0])
    return features



    







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

