"""
Implementation of the human and machine policies in the paper
"""
import datetime
import random
from numba import njit
import numpy as np
import math
from typing import Callable

from environments.env import Environment
from copy import copy


# TODO make sure policies sums to one!

    # class ConfidenceSet:
    #     def __init__(self, shape, delta, true_policy=None):
    #         self.history = np.zeros(shape=shape)
    #         self.true_policy = true_policy
    #         self.shape = shape
    #         self.beta = np.zeros(shape=self.shape[0: len(self.shape) - 1]) if true_policy is not None else None
    #         self.empirical = self.true_policy
    #         self.delta = delta

    #     def update(self, tup):
    #         self.history[tup] += 1

    #     def get_empirical(self):
    #         if self.true_policy is not None:
    #             return self.empirical
    #         arr = np.copy(self.history)
    #         ax = len(self.shape) - 1
    #         arr[arr.sum(axis=ax) == 0] = 1
    #         self.empirical = arr / np.expand_dims(arr.sum(axis=ax), axis=ax)
    #         return self.empirical

    #     def get_beta(self, t_k):
    #         if self.true_policy is not None:
    #             return self.beta
    #         ax = len(self.shape) - 1
    #         ret = self.history.sum(axis=ax)
    #         ret[ret == 0] = 1
    #         w = 1
    #         for j in range(ax):
    #             w *= self.shape[j]
    #         m = self.shape[ax]
    #         self.beta = np.sqrt(2 * (7 * math.log(t_k) + math.log(w/self.delta) + math.log(2) * (m + 1)) / ret)
    #         return self.beta


class Agent:
    """
    Agent superclass
    """
    def __init__(self):
        self.policy = {}

    def update_obs(self, *args):
        """Add observations to the record"""

    def update_policy(self, *args):
        """Update the action policy"""

    def take_action(self, *args):
        """Return an action based on the policy"""

class MachineDriverAgent(Agent):
    def __init__(self):
        """Initialize network and metrics"""
        super().__init__()

    def update_obs(self, *args):
        """Return input batch  for training"""
        pass

    def update_policy(self, *args):
        """Implement train step """
        pass 

    def take_action(self, *args):
        """Return an action based on the policy"""
        pass 


class NoisyDriverAgent(Agent):
    def __init__(self, env: Environment, noise_sd: float, noise_sw=.0):
        """
        A noisy driver, which chooses the cell with the lowest noisy estimated cost.

        Parameters
        ----------
        env: Environment

        noise_sd : float
            Standard deviation of the Gaussian noise (i.e., `N(0, noise_sd)`)
        
        noise_sw : float
            Standard deviation of the Gaussian noise beacuse of switching from Machine to Human
        """
        super().__init__()
        self.noise_sd = noise_sd
        self.noise_sw = noise_sw
        self.type_costs = env.type_costs


    def take_action(self, curr_state):
        '''
        current state in form of  ['road', 'no-car','car','road','car', ...]
        human considers only next row no the others
        ''' 
          
        noisy_next_cell_costs = [self.type_costs[nxt_cell_type] + random.gauss(0,self.noise_sd) + random.gauss(0, self.noise_sw) if nxt_cell_type!='wall' else np.inf for nxt_cell_type in curr_state[2:5]]
        min_estimated_cost = np.min(noisy_next_cell_costs)
        # ties are broken randomly
        possible_actions = np.argwhere(noisy_next_cell_costs == min_estimated_cost)
        n_possible_actions = possible_actions.shape[0]
        action = random.choices(possible_actions, [1/n_possible_actions]*n_possible_actions)[0]
        
        return action



