"""
Implementation of various switching agents 
"""
import numpy as np
import copy
import random
from torch import nn
import torch

from agent.agents import  Agent
from networks.networks import CriticNet, OptionCriticNet
from environments.utils_env import state2features
from environments.env import Environment




class FixedSwitchingHuman(Agent):
    """
    Switching policy chooses always human
    """
    def __init__(self):
        super(FixedSwitchingHuman, self).__init__()
        self.trainable = False
        
    def take_action(self, state):
        return 0

class FixedSwitchingMachine(Agent):
    """
    Switching policy chooses always machine
    """
    def __init__(self, n_state_features, optimizer):
        """Initialize network and hyperparameters"""
        super(FixedSwitchingMachine, self).__init__()
        self.network = CriticNet(n_state_features[1])
        self.optimizer = optimizer(self.network.parameters())
        self.n_state_features = n_state_features[0]
        self.trainable = True


    def update_obs(self, *args):
        """Return input batch  for training"""
        pass

# Need to use njit decorator?
    def update_policy(self, weighting, td_error):
        """
        Implement train step 

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = F_t * rho_t, for on-policy weighting = 1

        td_error: torch.LongTensor
            TD_error c + V(s+1)  - V(s)
        """
        # weighting and c'(s,a) + V(s+1) must have been computed with torch.no_grad()
        # maybe weighting needs clamp(0,1)!!!
        v_loss = td_error.pow(2).mul(0.5).mul(weighting)
        # TODO: v_loss = v_loss.mean() for batch update

        self.optimizer.zero_grad()
        v_loss.backward()
        self.optimizer.step()

    def take_action(self, curr_state):
        return 1


class SwitchingAgent(Agent):
    """
    Switching policy chooses always machine
    """
    def __init__(self, n_state_features, optimizer, c_M, c_H, eps=.01):
        """Initialize network and hyperparameters"""
        super(SwitchingAgent, self).__init__()
        # TODO: change  n_state_features+1 for 1-hot encoding  
        self.network = OptionCriticNet(n_state_features[1]+2,c_M,c_H)
        self.optimizer = optimizer(self.network.parameters())
        self.epsilon = eps
        self.trainable = True
        self.n_state_features = n_state_features[0]


    def update_obs(self, *args):
        """Return input batch  for training"""
        pass

# Need to use njit decorator?
    def update_policy(self, weighting, td_error):
        """
        Implement train step 

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = F_t * rho_t, for on-policy weighting = 1

        td_error: torch.LongTensor
            TD_error c + switch(s+1)*Q(s+1, M) + (1-switch(s+1))*Q(s+1, H)
            - (switch(s)*Q(s, M) + (1-switch(s))*Q(s, H))
        """
        # weighting and c'(s,a) + V(s+1) must have been computed with torch.no_grad()
        # maybe weighting needs clamp(0,1)!!!
        v_loss = td_error.pow(2).mul(0.5).mul(weighting)
        # TODO: v_loss = v_loss.mean() for batch update
        self.optimizer.zero_grad()
        v_loss.backward()
        # nn.utils.clip_grad_norm_(self.network.parameters(), 0.8)
        self.optimizer.step()

    def take_action(self, curr_state):
        """
        Return the switching decision given the current state 

        Parameters
        ----------
        curr_state: list of strings
            Current state vector 
        
        Returns
        -------
        switch: int
            The switching decision
        """
        state_feature_vector  = state2features(curr_state, self.n_state_features)
        # TODO: change human/machine feauture value for 1-hot encoding
         
        human_option_value = self.network(state_feature_vector + [0.,1.]).detach().item()
        machine_option_value = self.network(state_feature_vector + [1.,0.]).detach().item()
        # epsilon greedy
        p = random.random()
        if p < 1- self.epsilon:
            switch = np.argmin([human_option_value, machine_option_value]).flatten()[0]
        else:
            switch = random.choices([0, 1], [.5, .5])[0]
        
        return switch 
