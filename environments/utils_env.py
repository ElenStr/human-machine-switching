"""
Environment help functions.
"""

import numpy as np
import random
USE_GRASS_OBSTACLE = False


def general_grid(cells, start,finish, env):
    
    grass_obstacle = USE_GRASS_OBSTACLE and random.random() <.5
    for row in range(start,finish):
        for col in range(env.width):
            cell_probs = list(env.type_probs[env.traffics[row]].values())
            cells[col, row] = random.choices(env.cell_types, cell_probs)[0]
            if grass_obstacle and cells[col, row] == 'grass':
            # remove grass cells since a grass obstacle will be added later
            # grass cells will become with 0.5 road and 0.5 stone
                cells[col, row] = 'road' if random.random() <.5 else 'stone'



    # Add random grass sequence in middle lane
    # pick end-start in [2, depth] if machine view is L/3 rows
    # choose start in [L/3(2?),L - depth - 1]
    height = finish - start
    if grass_obstacle:
        n_grass_cells = random.randint(2, env.depth)
        start = random.randint(height//3,height-env.depth - 1)
        for r in range(start, start+n_grass_cells):
            cells[1, r] = 'grass'
    
    

def two_lanes_obstcales(cells, start, finish, obstacle_type, start_obst_mid=False):
    for row in range(start, finish):
        
        if row==start:
            # TODO: works only for width=3
            cells[0, row] = 'road'
            cells[1, row] = obstacle_type if start_obst_mid else 'road'
            cells[2, row] = 'road' if start_obst_mid else obstacle_type
        else:
            cells[0, row] = obstacle_type
            cells[1, row] = obstacle_type
            cells[2, row] = 'road'

    
        
            
def difficult_grid(cells, start, finish, obstacle_type, start_obst_mid=False):
    
    # TODO: works only for width=3
    cells[0, start] = 'road'
    cells[1, start] = obstacle_type if start_obst_mid else 'road'
    cells[2, start] = 'road' if start_obst_mid else obstacle_type
    for row in range(start+1,finish, step=2):        
        cells[0, row] = obstacle_type
        cells[1, row] = obstacle_type
        cells[2, row] = 'road'
        cells[0, row+1] = obstacle_type
        cells[1, row+1] = 'road'
        cells[2, row+1] = obstacle_type

        
