import random
import numpy as np

# from plot.plot_path import PlotPath

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

USE_GRASS_OBSTACLE = True

class GridWorld:
    """
    Grid world with height, width, cell types, and traffic levels
    """

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
        self.cell_types = cell_types
        self.traffic_levels = traffic_levels
        self.type_costs = type_costs

        self.current_coord = ((self.width - 1) // 2, 0)

        self.depth = depth

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
        if x==0:
            nxt = 2
            state.append(self.traffic_levels[y+1])
            state.append('wall')
            for r in range(2):
                state.append(self.cell_types[r, y+i])
        elif x==self.width - 1:
            nxt = 2
            state.append(self.traffic_levels[y+1])
            for r in range(1,3):
                state.append(self.cell_types[r, y+i])
            state.append('wall')
        
        for i in range(nxt,depth+1):
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


class Environment:
    """
    Lane driving environment in the paper
    """

    def __init__(self):

        self.traffic_probs = TRAFFIC_LEVEL_PROBS
        self.type_probs = TYPE_PROBS
        self.cell_types = CELL_TYPES
        self.traffic_levels = TRAFFIC_LEVELS
        self.type_costs = TYPE_COSTS

    def generate_grid_world(self, width, height, init_traffic_level: str):
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

        # DONE: choose everything but grass some times
        grass_obstacle = USE_GRASS_OBSTACLE and random.random() <.5
        for row in range(height):
            traffic_probs = list(self.traffic_probs[traffics[row]].values())
            traffics.append(random.choices(self.traffic_levels, traffic_probs)[0])
            for col in range(width):
                cell_probs = list(self.type_probs[traffics[row]].values())
                cells[col, row] = random.choices(self.cell_types, cell_probs)[0]
                
                if grass_obstacle and cells[col, row] == 'grass':
                # remove grass cells since a grass obstacle will be added later
                # grass cells will become with 0.5 road and 0.5 stone
                    cells[col, row] = 'road' if random.random() <.5 else 'stone'



        # Add random grass sequence in middle lane
        # pick end-start in [L/6, L/3] if machine view is L/3 rows
        # choose start in [L/3(2?),L]
                
        if grass_obstacle:
            n_grass_cells = random.randint(height//6, height//3)
            start = random.randint(height//3,height-1)
            for r in range(start, start+n_grass_cells):
                cells[1, r] = 'grass'


        middle_width = width // 2
        cells[middle_width, 0] = 'road'
        grid_world = GridWorld(width, height, cells, traffics, self.type_costs)
        return grid_world