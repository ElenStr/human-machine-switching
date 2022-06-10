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
from environments.taxi_env import MapEnv




class FixedSwitchingHuman(Agent):
    """
    Switching policy chooses always human
    """
    def __init__(self):
        super(FixedSwitchingHuman, self).__init__()
        self.trainable = False
        
    def take_action(self, state, train, online,use_target=False):
        return 0

class FixedSwitchingMachine(Agent):
    """
    Switching policy chooses always machine
    """
    def __init__(self, n_state_features, optimizer, c_M, batch_size=1):
        """Initialize network and hyperparameters"""
        super(FixedSwitchingMachine, self).__init__()
        # Critic need two extra features for the agent 
        self.network = CriticNet(n_state_features+2, c_M)
        self.target_network = CriticNet(n_state_features+2, c_M)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_update_freq = 5000
        self.timesteps = 0
        self.optimizer = optimizer(self.network.parameters())
        self.n_state_features = n_state_features
        self.F_t= np.zeros(batch_size)        
        self.var_rho = np.ones(batch_size)
        self.trainable = True


    def update_obs(self, *args):
        """Return input batch  for training"""
        pass

    def update_policy(self, weighting, td_error, do_update=True):
        """
        Implement train step 

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = F_t * rho_t, for on-policy weighting = 1

        td_error: torch.LongTensor
            TD_error c(s,a) + V(s+1)  - V(s)
        """
        if self.timesteps % self.target_update_freq ==0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.timesteps+=1
        v_loss = td_error.pow(2).mul(0.5).mul(weighting)
        v_loss = v_loss.mean()

        self.optimizer.zero_grad()
        v_loss.backward()
        # nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()
        # print(f"Step:{self.timesteps}")


    def take_action(self, curr_state, train, online=False, use_target=False):
        return 1


class SwitchingAgent(Agent):
    """
    Switching policy chooses always machine
    """
    def __init__(self, n_state_features, optimizer, c_M, c_H, eps_fn, batch_size=1):
        """Initialize network and hyperparameters"""
        super(SwitchingAgent, self).__init__()
       
        self.network = OptionCriticNet(n_state_features+2,c_M,c_H)
        self.optimizer = optimizer(self.network.parameters())        
        self.target_network = OptionCriticNet(n_state_features+2, c_M,c_H)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_update_freq = 5000
        self.timesteps = 0
        self.epsilon_fn = eps_fn
        self.epsilon = self.epsilon_fn(0)      
        self.F_t= np.zeros(batch_size)
        self.var_rho = np.ones(batch_size)
        self.trainable = True
        self.n_state_features = n_state_features


    def update_obs(self, *args):
        """Return input batch  for training"""
        pass

    def update_policy(self, weighting, td_error):
        """
        Implement train step 

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = F_t * rho_t, for on-policy weighting = 1

        td_error: torch.LongTensor
            TD_error c(s,a) + switch(s+1)*Q(s+1, M) + (1-switch(s+1))*Q(s+1, H)
            - (switch(s)*Q(s, M) + (1-switch(s))*Q(s, H))
        """
        if self.timesteps % self.target_update_freq ==0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        self.epsilon = self.epsilon_fn(self.timesteps)         
        self.timesteps+=1
        v_loss = td_error.pow(2).mul(0.5).mul(weighting)       
        v_loss = v_loss.mean()
        self.optimizer.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()


    def take_action(self, curr_state, train=True, online=False, use_target=False):
        """Return the switching decision (0 or 1) given the current state"""
        
        network = self.network if not use_target else self.target_network
        state_feature_vector  = MapEnv.state2features(curr_state, self.n_state_features)
        human_option_value = network(state_feature_vector + [0.,1.]).detach().item()
        machine_option_value = network(state_feature_vector + [1.,0.]).detach().item()
        p = random.random()
        # epsilon greedy only when training or in online evaluation
        epsilon = self.epsilon if train or online else 0.0
        if p < 1- epsilon:
            switch =  int(human_option_value >= machine_option_value)
        else:
            switch = random.choices([0, 1], [.5, .5])[0]
        return switch 

