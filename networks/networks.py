import torch
from torch import nn, as_tensor
import torch.nn.functional as F

class Network(nn.Module):
    """ 1-layer architecture """
    def __init__(self, dim_in: int, dim_out: int):
        super(Network, self).__init__()
        #TODO hidden = f(dim_in)
        hidden = 64
        self.input_dim = dim_in
        self.inp_layer = nn.Linear(dim_in, hidden)
        self.fc_layer = nn.Linear(hidden, dim_out)

    def forward(self, features, activation):
        """ features is the featurized input vector"""         
        input = as_tensor(features, dtype=torch.float)
        # Padd input when approaching end of episode
        features_size = len(features)
        if features_size < self.input_dim:
            padd = torch.ones(self.input_dim-features_size, dtype=torch.float)
            input = torch.cat((input,padd) ) 
        output = self.fc_layer(F.relu(self.inp_layer(input)))
        return  activation(output)
    
    def initialize_layer(layer):
        pass

class ActorNet(Network):
    def __init__(self, n_features: int , n_actions: int ):
        """ 
        Initialization of actor architecture

        Parameters
        ----------
        n_features: int
            Dimension of input state feature vector
        n_actions: int
            Number of actions
        """
        super(ActorNet, self).__init__(n_features, n_actions)
    
    def forward(self, features):
        return super().forward(features, lambda inp :F.softmax(inp, dim=-1))

class CriticNet(Network):
    """ Standard critic """
    def __init__(self, n_features: int ):
        super(CriticNet, self).__init__(n_features, 1)
    
    def forward(self, features):
        return super().forward(features, lambda inp : inp)

class OptionCriticNet(Network):
    """ Option value critic """

    def __init__(self, n_features: int, c_c_M: float, c_c_H: float  ):
        """
        The network architecture for the option value function estimation
        
        Parameters
        ----------
        n_features: int
            Dimension of input feature vector = n_state_features + 1

        c_c_M: float
            Machine control cost

        c_c_H: float
            Human control cost        
        """
        super(OptionCriticNet, self).__init__(n_features, 1)
        self.c_M = c_c_M
        self.c_H = c_c_H
    
    def forward(self, features):
        """
        Parameters
        ----------
        features: int array-like
            The featurized state vector with appended the agent (Machine or Human) feature value            
        """
        agent_control_cost = self.c_M if features[-1] else self.c_H
        return super().forward(features, lambda inp : inp+agent_control_cost)

if __name__ == '__main__':
    from random import randint
    from copy import copy
    depth = 3
    n_row_features = 4
    n_state_features = 1 + 3*4
    n_actions = 3
    c_M = 0.3
    c_H = 0

    actor_net = ActorNet(n_state_features, n_actions)
    critic_net = CriticNet(n_state_features)
    option_critic = OptionCriticNet(n_state_features+1, c_M, c_H)
    option_critic_no_control = copy(option_critic)
    option_critic_no_control.c_M = 0.

    state_inp = [randint(0,1)*1.]*n_state_features
    option_inp = state_inp+[randint(0,1)*1.]

    print(actor_net(state_inp))
    print(critic_net(state_inp))
    print(option_critic(option_inp))
    print(option_critic_no_control(option_inp))
