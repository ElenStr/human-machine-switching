import torch
from torch import nn, as_tensor
from torch.nn import functional as F 
import numpy as np

def initialize_layer(layer,w_scale=1.):
    nn.init.uniform_(layer.weight.data)
    # nn.init.constant_(layer.bias.data, 0)
    return layer

class Network(nn.Module):
    """ 1-layer architecture """
    def __init__(self, dim_in: int, dim_out: int):
        super(Network, self).__init__()
        hidden = int(2**(np.ceil(np.log2(dim_in)) + 1))
        self.input_dim = dim_in
        self.batchnorm = nn.BatchNorm1d(dim_in, affine=False, track_running_stats=False)
        self.inp_layer = initialize_layer(nn.Linear(dim_in, hidden))
        self.fc_layer = initialize_layer(nn.Linear(hidden, dim_out))

    def forward(self, features, activation):
        """ features is the featurized input vector"""         
        input = as_tensor(features, dtype=torch.float)
        input = torch.squeeze(self.batchnorm(torch.unsqueeze(input,1)))
        x = self.inp_layer(input)

        # print(x) 

        # x = F.relu(x)
        x = self.fc_layer(x)
        # print(x) 

        output = activation(x)
        return  output
    
    

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
        print()
        super(ActorNet, self).__init__(n_features, n_actions)
        
         
    
    def forward(self, features):
        return super().forward(features, lambda inp : inp)

class CriticNet(Network):
    """ Standard critic """
    def __init__(self, n_features: int, c_M ):
        super(CriticNet, self).__init__(n_features, 1)
        self.c_M = c_M
        self.needs_agent_feature = True
    
    def forward(self, features):
        return super().forward(features, lambda inp : inp + self.c_M)

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
        self.needs_agent_feature = True
    
    def forward(self, features):
        """
        Parameters
        ----------
        features: int array-like
            The featurized state vector with appended the agent (Machine or Human) feature value            
        """
        features = np.array(features).squeeze()
        if  len(features.shape)==1:
            agent_control_cost = self.c_M if features[-2] else self.c_H
        else:
            # needed when batch_size > 1
            agent_control_cost = self.c_M*features[:,-2] + self.c_H*features[:,-1]
        return super().forward(features, lambda inp : inp+agent_control_cost)



