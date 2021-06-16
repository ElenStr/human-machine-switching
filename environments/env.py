import random
import numpy as np

from plot.plot_path import HUMAN_COLOR, MACHINE_COLOR
from environments.utils_env import *

CELL_TYPES = ['road', 'grass', 'stone', 'car']
TRAFFIC_LEVELS = ['no-car', 'light', 'heavy']

TYPE_PROBS = {
    'no-car': {'road': 0.7, 'grass': 0.2, 'stone': 0.1, 'car': 0},
    'light': {'road': 0.6, 'grass': 0.2, 'stone': 0.1, 'car': 0.1},
    'heavy': {'road': 0.5, 'grass': 0.2, 'stone': 0.1, 'car': 0.2}
}
#DONE: change 
TRAFFIC_LEVEL_PROBS = {
    'no-car': {'no-car': 0.8, 'light': 0.2, 'heavy': 0},
    'light': {'no-car': 0.2, 'light': 0.6, 'heavy': 0.2},
    'heavy': {'no-car': 0, 'light': 0.2, 'heavy': 0.8},
}

# default costs
TYPE_COSTS = {'road': 0, 'grass': 2, 'stone': 4, 'car': 10}


class GridWorld:
    """
    Grid world with height, width, cell types, and traffic levels
    """
    width_st=3
    def __init__(self, width, height, cell_types: dict, traffic_levels: list, type_costs: dict, depth: int = 3):
        """

        Parameters
        ----------
        height : int
            Height of the grid env

        width : int
            Width of the grid env

        cell_types : dict
            Type of each grid cell with coordinate (x, y), i.e., {(x, y): type}.

        traffic_levels: list
            Traffic level of each row

        type_costs: dict
            Cost of each cell type

        depth: int
            The number of next rows in the state vector
        """
        self.height = height
        self.width = width
        GridWorld.width_st = self.width
        self.cell_types = cell_types
        self.traffic_levels = traffic_levels
        self.type_costs = type_costs

        self.current_coord = ((self.width - 1) // 2, 0)

        self.depth = depth

    def next_coords(self, x,y, action):
        # the new cell after taking 'action'
        x_n, y_n = x + action - 1, y + 1

        # TODO: add termination when hitting a car 
        # maybe better in step function
        # the top row
        if y_n >= self.height:
            y_n = y

        # handle wall
        is_wall = x_n < 0 or x_n >= self.width
        if is_wall:
            if x_n < 0:
                x_n = 0
            else:
                x_n = self.width - 1

        next_coord = (x_n, y_n)
      
        return next_coord
    def next_cell(self, action: int, move: bool =True):
        """
        Coordinates of the next cell if action is taken

        Parameters
        ----------
        action : int
            The action to be taken
        move : bool
            Move to next cell if true

        Returns
        -------
        next_coord : int
            The new coordinates
        finished : bool
            If time exceeds the episode length
        """

        x, y = self.current_coord

        # the new cell after taking 'action'
        x_n, y_n = x + action - 1, y + 1

        # TODO: add termination when hitting a car 
        # maybe better in step function
        finished = False
        # the top row
        if y_n >= self.height:
            finished = True
            y_n = y

        # handle wall
        is_wall = x_n < 0 or x_n >= self.width
        if is_wall:
            if x_n < 0:
                x_n = 0
            else:
                x_n = self.width - 1

        next_coord = (x_n, y_n)
      
        if move:
            self.current_coord = next_coord

        return next_coord, finished

    def coords2state(self, x,y):
        state = [self.cell_types[x,y]]       
        nxt = 1
        # Add wall type if current cell is leftmost (rightmost)
        # if not in last row
        if x==0 and y<self.height-1:
            nxt = 2
            state.append(self.traffic_levels[y+1])
            state.append('wall')
            for r in range(2):
                state.append(self.cell_types[r, y+1])
        elif x==self.width - 1 and y<self.height-1 :
            nxt = 2
            state.append(self.traffic_levels[y+1])
            for r in range(1,3):
                state.append(self.cell_types[r, y+1])
            state.append('wall')
        # state includes min(depth, remaining rows ahead )
        upper_bound = min(self.depth, self.height-y -1) +1
        for i in range(nxt,upper_bound):
            state.append(self.traffic_levels[y+i])            
            for r in range(3):
                state.append(self.cell_types[r, y+i])
        return state

    def current_state(self):
        """
        Returns the current state of the MDP in the form of:

        [current_cell_type, features_1, features_2, features_3,..,features_depth ],
        where features_i is :

        traffic_factor, left_cell_type, mid_cell_type, right_cell type            
        for the i-th next row

        """
        x, y = self.current_coord
        state = [self.cell_types[x,y]]       
        nxt = 1
        # Add wall type if current cell is leftmost (rightmost)
        # if not in last row
        if x==0 and y<self.height-1:
            nxt = 2
            state.append(self.traffic_levels[y+1])
            state.append('wall')
            for r in range(2):
                state.append(self.cell_types[r, y+1])
        elif x==self.width - 1 and y<self.height-1 :
            nxt = 2
            state.append(self.traffic_levels[y+1])
            for r in range(1,3):
                state.append(self.cell_types[r, y+1])
            state.append('wall')
        # state includes min(depth, remaining rows ahead )
        upper_bound = min(self.depth, self.height-y -1) +1
        for i in range(nxt,upper_bound):
            state.append(self.traffic_levels[y+i])            
            for r in range(3):
                state.append(self.cell_types[r, y+i])
        return state

    def step(self, action):
        """   Next state of the MDP  """
        next_coord, finished = self.next_cell(action)
        next_state = self.current_state()
        # Cost recieved when we enter the state
        cost = self.type_costs[next_state[0]]
        return next_state, cost, finished

    def reset(self):
        """ resets the trajectory """
        self.current_coord = ((self.width - 1) // 2, 0)

    def plot_trajectory(self, switching_agent, acting_agents, plt_path, show_cf=False, machine_only=False):
        """
        Plot trajectory of agent in grid environment.

        Parameters
        ----------
        switching_agent:  Agent
            The switching agent

        acting_agents:  list of Agent
            The actings agents

        plt_path: PlotPath
            The plotter object with the trajectory to be plotted
        
        show_cf: bool
            Wether to plot human counterfactual decision when switching chose machine

        machine_only: bool
            True if the swithing agent chooses always machine
        
        """       

        human_cf_lines = []
        human_cf_costs = []              
        d_tminus1 = 0
        timestep = 0
        trajectory_cost = 0
        machine_picked = 0
        self.reset()
        while True:
            timestep+=1          
           
            current_state = self.current_state()
            src = self.current_coord

            d_t = switching_agent.take_action(current_state, False)
            option = acting_agents[d_t] 
            machine_picked+=d_t
            if not d_t:  
                action = option.take_action(current_state, d_tminus1)
            else:
                action, policy = option.take_action(current_state)
                if show_cf:
                    for key in range(len(human_cf_lines)):
                        cf_src =  human_cf_lines[key][-1][1]
                        cf_state = self.coords2state(cf_src[0], cf_src[1])
                        cf_action =  acting_agents[0].take_action(cf_state)
                        cf_dst = self.next_coords(cf_src[0], cf_src[1], cf_action)
                        human_cf_lines[key].append((cf_src, cf_dst))
                        human_cf_costs[key]+=(self.type_costs[self.cell_types[cf_dst]])

                    if (not machine_only) or (machine_only and timestep==1):# record here human alternative
                        human_only_action = acting_agents[0].take_action(current_state)
                        human_only_dst = self.next_cell(human_only_action, move=False)[0]
                        human_cf_lines.append([(src, human_only_dst)])
                        human_cf_costs.append(trajectory_cost + self.type_costs[self.cell_types[human_only_dst]])
                            
            d_tminus1 = d_t
            
            next_state, cost, finished = self.step(action)
            if finished:
                break 
            dst = self.current_coord

            c_tplus1 = cost + option.control_cost         
                        
            trajectory_cost += c_tplus1            

            clr = MACHINE_COLOR if d_t else HUMAN_COLOR
            plt_path.add_line(src, dst, clr)
    
        if human_cf_costs:
            key = np.argmin(human_cf_costs)
            for src, dst in human_cf_lines[key]:
                plt_path.add_line(src, dst, HUMAN_COLOR)        
                        

        return trajectory_cost, machine_picked/timestep

class Environment:
    """
    Lane driving environment in the paper
    """
    width = 3
    def __init__(self):

        self.traffic_probs = TRAFFIC_LEVEL_PROBS
        self.type_probs = TYPE_PROBS
        self.cell_types = CELL_TYPES
        self.traffic_levels = TRAFFIC_LEVELS
        self.type_costs = TYPE_COSTS

    def n_state_strings(self, depth, width):
        """State size with string features"""
        return 1 + depth*(width + 1)
    
    def n_state_one_hot(self, depth, width):
        """State size in 1-hot encoding"""
        n_cell_types = len(self.cell_types)
        n_traffic_levels = len(self.traffic_levels)
        n_state_features_1hot =  n_cell_types + depth*( n_traffic_levels + 1 + width*(n_cell_types + 1))
        return n_state_features_1hot

    def generate_grid_world(self, width, height, init_traffic_level: str, scerario_fn=lambda c,s,f:two_lanes_obstcales(c,s,f,'grass'), depth=3):
        """
        Assign each cell a type (i.e., 'road', 'grass', 'stone', or 'car')
        independently at random based on the traffic level.

        Parameters
        ----------
        width : int
            Width of the grid env
        height : int
            Height of the grid env
        init_traffic_level: str
            Initial traffic level
        Returns
        -------
        grid_world: GridWorld
        """
        traffics = [init_traffic_level]
        cells = {}
        Environment.width = width
        for row in range(height):       
            traffic_probs = list(self.traffic_probs[traffics[row]].values())
            traffics.append(random.choices(self.traffic_levels, traffic_probs)[0])
        self.traffics = traffics
        self.depth = depth
        general_grid(cells,0,height//2 -1, self)
        scerario_fn(cells, height//2 -1,height)

        middle_width = width // 2
        cells[middle_width, 0] = 'road'
        grid_world = GridWorld(width, height, cells, traffics, self.type_costs, depth)
        return grid_world   

    @staticmethod
    def feature2net_input(value, n_onehot):
        """
        Parameters
        ----------
        value: int 
            Scalar feature value

        n_onehot: int
            Number of possible feature values

        Returns
        -------
        f_v: list
            The onehot represantation of value
        """
        f_v = [0.]*n_onehot
        f_v[n_onehot - value - 1] = 1.
        return f_v
    @staticmethod
    def agent_feature2net_input(value):
        return Environment.feature2net_input(value,2)
    
    @staticmethod
    def state2features(state, n_features, real_v=False):
        """
        Parameters
        ----------
        state: list of strings
            Current cell type and traffic factor and cell types of next rows 

        n_features: int
            Number of features required by the network
        
        real_v: bool
            Return features as real values

        Returns
        -------
        features: list of int
            The state feature vector
        """
        

        cell_t = np.array(CELL_TYPES+['wall'])
        traffic_l = np.array(TRAFFIC_LEVELS)
        features = []
        feature = np.argwhere(cell_t == state[0])[0][0]
        # TODO: onehot encoding for features
        if real_v:
            features.append((feature + 1)*0.2)
        else:
            features.extend(Environment.feature2net_input(feature, cell_t.size - 1))

        for i in range(1,n_features):
            if (i-1) % (Environment.width+1) == 0:
                state_i =  state[i] if i <= len(state)-1 else 'not-available'
                feature = np.argwhere(traffic_l == state_i)
                # add value for no traffic for last states
                feature = traffic_l.size if not feature.size else feature[0][0]
                # 1 stands for not available
                if real_v: 
                    features.append(feature + 2.)
                else:
                    features.extend(Environment.feature2net_input(feature, traffic_l.size + 1))              


            else:
                state_i =  state[i] if i <= len(state)-1 else 'wall'
                feature = np.argwhere(cell_t == state_i)[0][0]
                if real_v:
                    features.append((feature + 1.) * 0.2)
                else:
                    features.extend(Environment.feature2net_input(feature, cell_t.size)) 


        return features
