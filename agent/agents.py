"""
Implementation of the human and machine policies in the paper
"""
import random
import numpy as np
import math
import torch
from collections import defaultdict

from environments.env import Environment, GridWorld
from environments.utils_env import state2features
from networks.networks import ActorNet
from copy import copy



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
    def __init__(self, n_state_features, n_actions, optimizer, c_M=0., entropy_weight=0.01, batch_size=1):
        """Initialize network and hyperparameters"""
        super(MachineDriverAgent, self).__init__()
        self.network = ActorNet(n_state_features[1], n_actions)
        self.optimizer = optimizer(self.network.parameters())
        self.entropy_weight = entropy_weight
        self.entropy_weight_0 = entropy_weight
        self.timestep = 0
        self.entropy_decay_freq = 5000
        self.control_cost = c_M
        self.trainable = True
        self.M_t = np.zeros(batch_size)
        self.n_state_features = n_state_features[0]


    def update_obs(self, *args):
        """Return input batch  for training"""
        pass

    def update_policy(self, weighting, delta, log_pi, entropy, do_update=True):
        """
        Implement train step 

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = M_t * rho_t, for on-policy weighting = switch(s) 

        delta: torch.LongTensor
            For off-policy delta = TD_error, for on-policy delta = v(s)

        current_policy: Categorical
            The current action policy distribution
        
        action: int
            The action taken
        """
        if self.timestep % self.entropy_decay_freq ==0 and self.entropy_weight >0.0:
            
            self.entropy_weight =self.entropy_weight_0* np.exp(-(self.timestep//20)//self.entropy_decay_freq)
        self.timestep+=1
        # weighting and delta must have been computed with torch.no_grad()
        # log_pi =  current_policy.log_prob(action)
        # # TODO: entropy = entropy.mean() for batch update 
        # entropy = current_policy.entropy()
        policy_loss = weighting * delta * log_pi  + self.entropy_weight*entropy
        # TODO: policy_loss = policy_loss.mean() for batch update
        
        policy_loss = policy_loss.mean()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
        
        self.optimizer.step()



    def take_action(self, curr_state):
        """
        Return an action given the current based on the policy 

        Parameters
        ----------
        curr_state: list of strings
            Current state vector 
        
        Returns
        -------
        action: int
            The action to be taken
        policy: Categorical
            The action policy distribution given form the network
        """
        # TODO: make machine worse than human+machine e.g. same feature value for road-stone
        state_feature_vector  = state2features(curr_state, self.n_state_features)
        actions_logits = self.network(state_feature_vector)
        policy = torch.distributions.Categorical(logits=actions_logits)
        action = policy.sample().item()
        return action, policy  

# needed to pickle human
def dd_init():
    return [0]*3

class NoisyDriverAgent(Agent):
    def __init__(self, env: Environment, noise_sd: float, noise_sw=.0, c_H=0.):
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
        super(NoisyDriverAgent, self).__init__()
        self.noise_sd = noise_sd
        self.noise_sw = noise_sw
        self.type_costs = env.type_costs
        self.control_cost = c_H
        self.trainable = False
        
        self.policy_approximation = defaultdict(dd_init)

    def update_policy(self, state, action):
        """Update policy approximation, needed for the off policy stage"""
        # The human action in reality depends only on next row
        human_obs = tuple(state[2:5] )
        self.policy_approximation[human_obs][action]+=1
            
    def get_policy_approximation(self, state, action):
        """ The approximated action policy distribution given the state """
        human_obs = tuple(state[2:5] )
        total_state_visit = sum(self.policy_approximation[human_obs])
        p_human_a_s = self.policy_approximation[human_obs][action] / total_state_visit
        return p_human_a_s


    def take_action(self, curr_state, switch=False):
        '''
        current state in form of  ['road', 'no-car','car','road','car', ...]
        human considers only next row, not the others
        ''' 
        
        switch_noise = self.noise_sw if switch else 0.  
        noisy_next_cell_costs = [self.type_costs[nxt_cell_type] + random.gauss(0,self.noise_sd) + random.gauss(0, switch_noise) if nxt_cell_type!='wall' else np.inf for nxt_cell_type in curr_state[2:5]]
        # if end of episode is reached
        if not noisy_next_cell_costs:
            return random.randint(0,2)
        min_estimated_cost = np.min(noisy_next_cell_costs) 
        # ties are broken randomly
        possible_actions = np.argwhere(noisy_next_cell_costs == min_estimated_cost).flatten()
        n_possible_actions = possible_actions.size
        
        action = random.choices(possible_actions, [1/n_possible_actions]*n_possible_actions)[0]
        return action


class RandomDriverAgent(Agent):
    def __init__(self):
        """A random driver """
        super(RandomDriverAgent, self).__init__()
        
        self.trainable = False
        self.control_cost = 0.0
        
        self.policy_approximation = defaultdict(dd_init)

    def update_policy(self, state, action):
        """Update policy approximation, needed for the off policy stage"""
        # The human action in reality depends only on next row
        human_obs = tuple(state[2:5] )
        self.policy_approximation[human_obs][action]+=1
            
    def get_policy_approximation(self, state, action):
        """ The approximated action policy distribution given the state """
        human_obs = tuple(state[2:5] )
        total_state_visit = sum(self.policy_approximation[human_obs])
        p_human_a_s = self.policy_approximation[human_obs][action] / total_state_visit
        return p_human_a_s


    def take_action(self, curr_state, switch=False):
        
                
        action = random.choices(range(3), [1/3, 1/3, 1/3])[0]
        return action


class OptimalAgent():
    def __init__(self, env: GridWorld, control_cost):        
        self.env = env
        self.control_cost = control_cost
        self.p = np.zeros(shape=(self.env.width,self.env.height, 3, self.env.width,self.env.height)) 
        for y in range(self.env.height):
            for x in range(self.env.width):
                for a in range(3):
                    nxt_x,nxt_y = self.env.next_coords(x,y,a)
                    self.p[x,y,a,nxt_x,nxt_y] = 1.

        self.policy = self.val_itr()

    def take_action(self, time, coords):
        x,y = coords
        return random.choices(range(3), self.policy[time][x][y])[0]
    
    def eval(self, n_try=1, plt_path=None):
        total_cost = []
        for i in range(n_try):
            self.env.reset()
            traj_cost = 0
            time = 0
            while True:
                cur_coords = self.env.current_coord
                action = self.take_action(time, cur_coords)
                _, cost, finished = self.env.step(action)
                if finished:
                    break
                traj_cost+=cost + self.control_cost
                if plt_path is not None:
                    plt_path.add_line(cur_coords, self.env.current_coord, 'red')
            total_cost.append(traj_cost)
        
        return np.mean(total_cost)

   
    def val_itr(self):
        ep_l = self.env.height
        n_ac = 3
        # q_val[time][state][action]
        q_val = np.zeros(shape=(ep_l, self.env.width,self.env.height, n_ac))
        # q_min[time][state]
        q_min = np.zeros(shape=(ep_l + 1, self.env.width,self.env.height))

        # policy[time][state][action]
        policy = np.zeros(shape=(ep_l, self.env.width,self.env.height,  n_ac))

        for i in range(ep_l):
            t = ep_l - i - 1
            for y in range(self.env.height):
                for x in range(self.env.width):
                    for a in range(n_ac):
                        nxt_x,nxt_y = self.env.next_coords(x,y,a)
                        q_val[t][x][y][a] = self.env.type_costs[self.env.cell_types[nxt_x,nxt_y]] + np.sum(self.p[x,y,a]* q_min[t + 1])

                    best_actions = np.where(q_val[t][x][y] == np.min(q_val[t][x][y]))[0]
                    policy[t][x,y][best_actions] = 1 / len(best_actions)
                    q_min[t][x][y] = np.min(q_val[t][x][y])

        return policy