"""
Implementation of the human and machine policies in the paper
"""
from copy import copy
import random
import numpy as np
from environments.taxi_env import MapEnv, get_angle, get_distance
import networkx as nx
import math
import torch
from collections import defaultdict
from config import TRIPS
from environments.env import Environment, GridWorld
from networks.networks import ActorNet



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
    def __init__(self, n_state_features, n_actions, optimizer, setting=1, c_M=0., entropy_weight=0.01, batch_size=1):
        """Initialize network and hyperparameters"""
        super(MachineDriverAgent, self).__init__()
        # n_state_features is the state size
        self.network = ActorNet(n_state_features, n_actions)
        self.optimizer = optimizer(self.network.parameters())
        self.entropy_weight_0 = entropy_weight
        self.timestep = 0
        self.control_cost = c_M
        self.trainable = True
        self.M_t = np.zeros(batch_size)
        self.setting = setting
        self.n_state_features = n_state_features


    def update_obs(self, *args):
        """Return input batch  for training"""
        pass

    def update_policy(self, weighting, delta, log_pi, entropy, use_entropy=True):
        """
        Implement train step 

        Parameters
        ----------
        weighting: torch.LongTensor
            For off-policy weighting = M_t * rho_t, for on-policy weighting = switch(s) 

        delta: torch.LongTensor
            TD Error

        log_pi: torch.LongTensor
            The current action log probability
        
        entropy: torch.LongTensor
            The entropy of the current policy distribution
        """
            
        if use_entropy:
            self.timestep+=1
            self.entropy_weight = self.entropy_weight_0/self.timestep
        else:
            self.entropy_weight = 0
        policy_loss = weighting * delta * log_pi  + self.entropy_weight*entropy
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
        curr_state: list of angle distance pairs
            Current state vector 
        
        Returns
        -------
        action: int
            The action to be taken
       
        policy: Categorical
            The action policy distribution given form the network
        """
        set_curr_state = copy(curr_state)
        

        state_feature_vector  = MapEnv.state2features(set_curr_state, self.n_state_features)
        actions_logits = self.network(state_feature_vector)
        
        valid_action_logits = actions_logits      
        policy = torch.distributions.Categorical(logits=valid_action_logits)        
        # TODO: may need to renormalize
        action = policy.sample().item()
        

        return action , policy  


class ShortestPathAgent():
    """An agent that always chooses the optimal action"""
    def __init__(self, env: MapEnv):        
        self.env = env
        

    def take_action(self, curr_state):
        source = self.env.current_node
        target = self.env.dest_node
        shortest_path = nx.shortest_path(self.env.G, source=source, target=target, weight='length')
        # get next node in shortest path, shortest_path[0] == source
        next_node_id = shortest_path[1]
        next_node_distance = get_distance(self.G, next_node_id, target)
        next_node_angle = get_angle(self.env.G, next_node_id, target)

        angle_idx = np.argwhere(next_node_angle == np.array(self.env.neighbors_sorted)[:,0]).flatten()
        distance_idx = np.argwhere(next_node_distance == np.array(self.env.neighbors_sorted)[:,1]).flatten()

        action =  np.intersect1d(angle_idx,distance_idx)[0]
        return action

class HumanTaxiAgent(Agent):
    def __init__(self, env: MapEnv, c_H=0):
        super(HumanTaxiAgent, self).__init__()
        
        self.c_H = c_H
        # Set current area before episoe starts
        self.env = env

    def compute_policy(self):
        # Trips in area that pass from current node
        trips_cur_node_in_cur_area = np.zeros(len(self.env.neighbors_sorted))
        for i,n in enumerate(self.env.neighbors_sorted):
            for trip in self.env.areas[self.env.cur_area]:
                if trip in self.env.G.nodes[n]['trips']:
                    trips_cur_node_in_cur_area[i]+=1
        denom = (sum(trips_cur_node_in_cur_area)*self.env.areas_counts[self.env.cur_area])
        if denom == 0:
            print("0 denom", sum(trips_cur_node_in_cur_area),self.env.areas_counts[self.env.cur_area] )
            raise Exception
        self.policy = [ v/denom for v in trips_cur_node_in_cur_area]
        # Keep neighbors before action is taken 
        self.neighbors = self.env.neighbors_sorted

    def get_policy(self, action):
        return self.policy[action]

    def take_action(self):
        for i,n in enumerate(self.env.neighbors_sorted):
            if self.env.cur_trip in self.env.G.nodes[n]['trips']:
                return i
        self.compute_policy()
        neighbors_probs = []
        for n in self.env.neighbors_sorted:
            neighbors_probs.append(self.policy[n])
        
        action = random.choices(list(range(self.env.MAX_OUT_DEGREE)), neighbors_probs)[0]
        return action
                



        






class HumanDriverAgent(Agent):
    def __init__(self, env: MapEnv, c_H=0.0):
        """
        The taxi driver, which chooses the cell with the lowest perceived cost.

        Parameters
        ----------
        env: Environment

        c_H: int
            Human control cost

        """
        super(HumanDriverAgent, self).__init__()
        self.control_cost = c_H
        self.trainable = False
        self.create_areas(env)

        # self.areas_trips_counts = np.zeros()
        

    
        
            
    def get_policy_approximation(self, state, action, grid_id):
        """ The approximated action policy distribution given the state """
        human_obs = tuple(state)
        total_state_visit = sum(self.policy_approximation[grid_id,human_obs])
        p_human_a_s = self.policy_approximation[grid_id,human_obs][action] / total_state_visit
        return p_human_a_s
    
    def get_actual_policy(self, state, next_state):
        """The true human policy distribution"""
        greedy_cell = min(state[1:4], key=lambda x: self.type_costs[x])
        next_cell = next_state[0]
        is_greedy =   next_cell == greedy_cell        
        n_cell = 2 if 'wall' in state[1:4] else 3
        n_opt =  sum(1 for cell in state[1:4] if cell == greedy_cell) 
        if self.setting == 1:
            if is_greedy:
                return (1 - self.prob_wrong)/n_opt + self.prob_wrong/n_cell
            else:
                return self.prob_wrong/n_cell
        elif self.setting <7:
            n_road = sum(1 for cell in state[1:4] if cell == 'road')
            n_car = sum(1 for cell in state[1:4] if cell == 'car')
            if is_greedy:
                if next_cell == 'road':
                    mu_a_s =  (1 - self.p_ignore_car)*(1 - self.prob_wrong)/n_road + self.p_ignore_car*(1 - self.prob_wrong)/(n_car + n_road) + self.prob_wrong/n_cell
                    return mu_a_s
                elif next_cell == 'car':
                    return 1/n_car
                else:
                    if 'car' in state[1:4]:
                        return (1 - self.p_ignore_car)*(1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
                    else:
                        return (1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
            else:
                if next_cell =='car':
                    return self.p_ignore_car * (1 - self.prob_wrong)/(n_road +n_car) + self.prob_wrong/n_cell
                else:
                    return self.prob_wrong/n_cell
        else:
            n_road = sum(1 for cell in state[1:4] if cell == 'road')
            n_grass = sum(1 for cell in state[1:4] if cell == 'grass')
            if is_greedy:
                if next_cell == 'road':
                    mu_a_s =  (1 - self.p_ignore_grass)*(1 - self.prob_wrong)/n_road + self.p_ignore_grass*(1 - self.prob_wrong)/(n_grass + n_road) + self.prob_wrong/n_cell
                    return mu_a_s
                elif next_cell == 'grass':

                    return 1/n_grass
                else:
                    if 'grass' in state[1:4]:

                        return (1 - self.p_ignore_grass)*(1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
                    else:

                        return (1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
            else:
                if next_cell =='grass':

                    return self.p_ignore_grass * (1 - self.prob_wrong)/(n_road +n_grass) + self.prob_wrong/n_cell
                else:

                    return self.prob_wrong/n_cell 
                       

    def get_policy(self, state, action, grid_id, next_state):
        if self.actual:
            return self.get_actual_policy(state, next_state)
        else:
            return self.get_policy_approximation(state, action, grid_id)



    def take_action(self, curr_state, switch=False):
        '''
        current state in form of  ['road', 'no-car','car','road','car', ...]
        human considers only next row, not the others
        ''' 
        
        # if end of episode is reached
        if len(curr_state) < 4:            
            return random.randint(0,2)

        p_choose = random.random()
        p_ignore = random.random()
        curr_state_for_human = copy(curr_state)
        # ignore car when switching
        if self.setting >= 4:
            for i, cell_type in enumerate(curr_state[1:4]):                
                if cell_type == 'car' and switch:                    
                    curr_state_for_human[i+1] = 'road'
        if self.setting<6:
            for i, cell_type in enumerate(curr_state[1:4]):                
                if cell_type == 'car' and p_ignore < self.p_ignore_car:                    
                    curr_state_for_human[i+1] = 'road'
        if self.setting ==7:
            for i, cell_type in enumerate(curr_state[1:4]):                
                if cell_type == 'grass' :                    
                    curr_state_for_human[i+1] = 'road'

        noisy_next_cell_costs = [self.type_costs[nxt_cell_type] for nxt_cell_type in curr_state_for_human[1:4]]

        if p_choose < self.prob_wrong:
            if curr_state[1] == 'wall':
                action = random.choices(range(2), [1/2, 1/2])[0] + 1
            elif curr_state[3] == 'wall':
                action = random.choices(range(2), [1/2, 1/2])[0] 
            else:
                action = random.choices(range(3), [1/3, 1/3, 1/3])[0]
            return action

        min_estimated_cost = np.min(noisy_next_cell_costs) 
        # ties are broken randomly
        possible_actions = np.argwhere(noisy_next_cell_costs == min_estimated_cost).flatten()
        n_possible_actions = possible_actions.size
        action = random.choices(possible_actions, [1/n_possible_actions]*n_possible_actions)[0]

        
        return action
      



# needed to pickle human
def dd_init():
    return [0]*3

class NoisyDriverAgent(Agent):
    def __init__(self, env: Environment, prob_wrong: float, setting=1, c_H=0., p_ignore_car=0.5):
        """
        A noisy driver, which chooses the cell with the lowest perceived cost.

        Parameters
        ----------
        env: Environment

        prob_wrong : float
            Probability of picking action at random
        
        setting : int
            Experimental setting

        c_H: int
            Human control cost

        p_ignore_car: float
            Probalility of perceiving car as road.

        """
        super(NoisyDriverAgent, self).__init__()
        self.p_ignore_car = p_ignore_car
        self.p_ignore_grass = 1. #used only in setting 7
        self.prob_wrong = prob_wrong
        self.type_costs = { **env.type_costs, 'wall':np.inf}
        self.control_cost = c_H
        self.trainable = False
        self.setting = setting
        self.actual = True
        
        self.policy_approximation = defaultdict(dd_init)

    def update_policy(self, state, action, grid_id):
        """Update policy approximation, needed for the off policy stage"""
        # grid_id enables approximation of human policy per multiple grid rollouts
        human_obs = tuple(state)        
        self.policy_approximation[grid_id,human_obs][action]+=1
            
    def get_policy_approximation(self, state, action, grid_id):
        """ The approximated action policy distribution given the state """
        human_obs = tuple(state)
        total_state_visit = sum(self.policy_approximation[grid_id,human_obs])
        p_human_a_s = self.policy_approximation[grid_id,human_obs][action] / total_state_visit
        return p_human_a_s
    
    def get_actual_policy(self, state, next_state):
        """The true human policy distribution"""
        greedy_cell = min(state[1:4], key=lambda x: self.type_costs[x])
        next_cell = next_state[0]
        is_greedy =   next_cell == greedy_cell        
        n_cell = 2 if 'wall' in state[1:4] else 3
        n_opt =  sum(1 for cell in state[1:4] if cell == greedy_cell) 
        if self.setting == 1:
            if is_greedy:
                return (1 - self.prob_wrong)/n_opt + self.prob_wrong/n_cell
            else:
                return self.prob_wrong/n_cell
        elif self.setting <7:
            n_road = sum(1 for cell in state[1:4] if cell == 'road')
            n_car = sum(1 for cell in state[1:4] if cell == 'car')
            if is_greedy:
                if next_cell == 'road':
                    mu_a_s =  (1 - self.p_ignore_car)*(1 - self.prob_wrong)/n_road + self.p_ignore_car*(1 - self.prob_wrong)/(n_car + n_road) + self.prob_wrong/n_cell
                    return mu_a_s
                elif next_cell == 'car':
                    return 1/n_car
                else:
                    if 'car' in state[1:4]:
                        return (1 - self.p_ignore_car)*(1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
                    else:
                        return (1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
            else:
                if next_cell =='car':
                    return self.p_ignore_car * (1 - self.prob_wrong)/(n_road +n_car) + self.prob_wrong/n_cell
                else:
                    return self.prob_wrong/n_cell
        else:
            n_road = sum(1 for cell in state[1:4] if cell == 'road')
            n_grass = sum(1 for cell in state[1:4] if cell == 'grass')
            if is_greedy:
                if next_cell == 'road':
                    mu_a_s =  (1 - self.p_ignore_grass)*(1 - self.prob_wrong)/n_road + self.p_ignore_grass*(1 - self.prob_wrong)/(n_grass + n_road) + self.prob_wrong/n_cell
                    return mu_a_s
                elif next_cell == 'grass':

                    return 1/n_grass
                else:
                    if 'grass' in state[1:4]:

                        return (1 - self.p_ignore_grass)*(1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
                    else:

                        return (1 - self.prob_wrong)/n_opt  + self.prob_wrong/n_cell
            else:
                if next_cell =='grass':

                    return self.p_ignore_grass * (1 - self.prob_wrong)/(n_road +n_grass) + self.prob_wrong/n_cell
                else:

                    return self.prob_wrong/n_cell 
                       

    def get_policy(self, state, action, grid_id, next_state):
        if self.actual:
            return self.get_actual_policy(state, next_state)
        else:
            return self.get_policy_approximation(state, action, grid_id)



    def take_action(self, curr_state, switch=False):
        '''
        current state in form of  ['road', 'no-car','car','road','car', ...]
        human considers only next row, not the others
        ''' 
        
        # if end of episode is reached
        if len(curr_state) < 4:            
            return random.randint(0,2)

        p_choose = random.random()
        p_ignore = random.random()
        curr_state_for_human = copy(curr_state)
        # ignore car when switching
        if self.setting >= 4:
            for i, cell_type in enumerate(curr_state[1:4]):                
                if cell_type == 'car' and switch:                    
                    curr_state_for_human[i+1] = 'road'
        if self.setting<6:
            for i, cell_type in enumerate(curr_state[1:4]):                
                if cell_type == 'car' and p_ignore < self.p_ignore_car:                    
                    curr_state_for_human[i+1] = 'road'
        if self.setting ==7:
            for i, cell_type in enumerate(curr_state[1:4]):                
                if cell_type == 'grass' :                    
                    curr_state_for_human[i+1] = 'road'

        noisy_next_cell_costs = [self.type_costs[nxt_cell_type] for nxt_cell_type in curr_state_for_human[1:4]]

        if p_choose < self.prob_wrong:
            if curr_state[1] == 'wall':
                action = random.choices(range(2), [1/2, 1/2])[0] + 1
            elif curr_state[3] == 'wall':
                action = random.choices(range(2), [1/2, 1/2])[0] 
            else:
                action = random.choices(range(3), [1/3, 1/3, 1/3])[0]
            return action

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
        human_obs = tuple(state )
        self.policy_approximation[human_obs][action]+=1
            
    def get_policy_approximation(self, state, action):
        """ The approximated action policy distribution given the state """
        human_obs = tuple(state )
        total_state_visit = sum(self.policy_approximation[human_obs])
        p_human_a_s = self.policy_approximation[human_obs][action] / total_state_visit
        return p_human_a_s


    def take_action(self, curr_state, switch=False):
        
                
        action = random.choices(range(3), [1/3, 1/3, 1/3])[0]
        return action



   


class OptimalAgent():
    """An agent that returns an optimal planning given a specific episode"""
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