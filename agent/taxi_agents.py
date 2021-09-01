from agent.agents import Agent
from networks.networks import ActorNet
from environments.taxi_env import FeatureHandler

import numpy as np
import torch

from copy import copy


class MachineTaxiDriver(Agent):
    def __init__(self, feature_handler: FeatureHandler, optimizer, c_m: float = 0,
                 entropy_w0: float = 0.01, batch_size: int = 1):
        super().__init__()
        self.feature_handler = feature_handler

        self.network = ActorNet(self.feature_handler.feature_size, feature_handler.max_actions)
        self.optimizer = optimizer(self.network.parameters())
        self.entropy_w0 = entropy_w0

        self.timestep = 0
        self.control_cost = c_m
        self.trainable = True
        self.M_t = np.zeros(batch_size)

    def update_policy(self, weighting, delta, log_pi, entropy, use_entropy=True):
        """
        Implement train step

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = M_t * rho_t, for on-policy weighting = switch(s)

        delta: torch.LongTensor
            For off-policy delta = TD_error, for on-policy delta = v(s)
        """

        if use_entropy:
            self.timestep += 1
            entropy_weight = self.entropy_w0 / self.timestep
        else:
            entropy_weight = 0
        # weighting and delta must have been computed with torch.no_grad()
        policy_loss = weighting * delta * log_pi + entropy_weight * entropy
        policy_loss = policy_loss.mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)

        self.optimizer.step()

    def take_action(self, state: tuple):
        """
        Return an action given the current based on the policy

        Parameters
        ----------
        state: tuple
            (current node number, destination node number)

        Returns
        -------
        action: int
            The action to be taken
        policy: Categorical
            The action policy distribution given form the network
        """
        # TODO: make machine worse than human+machine e.g. same feature value
        state_feature_vector = self.feature_handler.state2feature(state)
        logits = self.network(state_feature_vector)
        true_n_actions = self.feature_handler.action_numbers(state)
        # action_probs = torch.tensor([1/true_n_actions for i in range(true_n_actions)])
        action_probs = torch.distributions.Categorical(logits=logits[0:true_n_actions]).probs
        if (action_probs < 1e-5).any():
            action_probs = action_probs.clamp(1e-5, 1 - 1e-5)

        action_probs = action_probs / action_probs.sum()

        valid_policy = torch.distributions.Categorical(probs=action_probs)
        action = valid_policy.sample().item()

        return action, valid_policy

