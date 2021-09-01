"""
Implementation of various switching agents
"""
import numpy as np
import random
from torch import nn
import torch

from agent.agents import Agent
from networks.networks import CriticNet, OptionCriticNet

from agent.taxi_agents import FeatureHandler


class SwitchingAgentAbstract(Agent):
    trainable = True

    def __init__(self):
        super().__init__()

    def take_action(self, state, use_target=False):
        pass


class FixedSwitchingHuman(SwitchingAgentAbstract):
    """
    Switching policy chooses always human
    """

    def __init__(self):
        super(FixedSwitchingHuman, self).__init__()
        self.trainable = False

    def take_action(self, state, use_target=False):
        return 0


class FixedSwitchingMachine(SwitchingAgentAbstract):
    """
    Switching policy chooses always machine
    """

    def __init__(self, feature_handler: FeatureHandler, optimizer, c_M, batch_size=1):
        """Initialize network and hyper-parameters"""
        super().__init__()
        self.network = CriticNet(feature_handler.feature_size + 2, c_M)
        self.target_network = CriticNet(feature_handler.feature_size + 2, c_M)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_update_freq = 200
        self.timesteps = 0
        self.optimizer = optimizer(self.network.parameters())
        self.F_t = np.zeros(batch_size)

        self.var_rho = np.zeros(batch_size)

    # Need to use njit decorator?
    def update_policy(self, weighting, td_error):
        """
        Implement train step

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = F_t * rho_t, for on-policy weighting = 1

        td_error: torch.LongTensor
            TD_error = c + V(s+1)  - V(s)
        """
        if self.timesteps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.timesteps += 1
        # weighting and c'(s,a) + V(s+1) must have been computed with torch.no_grad()
        # maybe weighting needs clamp(0,1)!!!
        v_loss = td_error.pow(2).mul(0.5).mul(weighting)
        # TODO: v_loss = v_loss.mean() for batch update
        v_loss = v_loss.mean()

        self.optimizer.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

    def take_action(self, state, use_target=False):
        return 1


class SwitchingAgent(Agent):
    """
    Switching policy
    """

    def __init__(self, feature_handler: FeatureHandler, optimizer, c_M, c_H, eps_fn, batch_size=1):
        """Initialize network and hyper-parameters"""
        super().__init__()
        self.feature_handler = feature_handler
        self.network = OptionCriticNet(self.feature_handler.feature_size + 2, c_M, c_H)
        self.optimizer = optimizer(self.network.parameters())
        self.target_network = OptionCriticNet(self.feature_handler.feature_size + 2, c_M, c_H)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_update_freq = 5000
        self.timesteps = 0
        self.epsilon_fn = eps_fn
        self.epsilon = self.epsilon_fn(0)

        self.F_t = np.zeros(batch_size)

        self.var_rho = np.zeros(batch_size)

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
        if self.timesteps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon = self.epsilon_fn(self.timesteps)

        self.timesteps += 1
        # weighting and c'(s,a) + V(s+1) must have been computed with torch.no_grad()
        # maybe weighting needs clamp(0,1)!!!
        v_loss = td_error.pow(2).mul(0.5).mul(weighting)

        v_loss = v_loss.mean()
        self.optimizer.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

    def take_action(self, state, use_target=False):
        """
        Return the switching decision given the current state

        Parameters
        ----------
        state:
            Current state

        Returns
        -------
        switch: int
            The switching decision
        """
        # start machine training in off policy
        # if train and any(self.F_t==0):
        #     return 1
        network = self.network if not use_target else self.target_network
        state_feature = self.feature_handler.state2feature(state)
        human_option_value = network(state_feature + [0., 1.]).detach().item()
        machine_option_value = network(state_feature + [1., 0.]).detach().item()
        p = random.random()
        # TODO: epsilon greedy only when training. NOT FOR NOW
        # epsilon = self.epsilon if train or online else 0.0
        epsilon = self.epsilon
        if p < 1 - epsilon:
            switch = int(human_option_value >= machine_option_value)
        else:
            switch = random.choices([0, 1], [.5, .5])[0]

        return switch