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
            cells[col, row] = random.choices(env.cell_types.values(), cell_probs)[0]
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
    for row in range(start+1,finish, 2):        
        cells[0, row] = obstacle_type
        cells[1, row] = obstacle_type
        cells[2, row] = 'road'
        cells[0, row+1] = obstacle_type
        cells[1, row+1] = 'road'
        cells[2, row+1] = obstacle_type


def clean_grass(cells, start,finish, env):
    
    for row in range(start,finish):
        for col in range(env.width):
            env_cell_types = ['road', 'grass', 'stone']
            cell_probs = [0.5, 0.4, 0.1]
            cells[col, row] = random.choices(env_cell_types, cell_probs)[0]
                    
def clean_car(cells, start,finish, env):
    
    for row in range(start,finish):
        for col in range(env.width):
            env_cell_types = ['road', 'car', 'stone']
            cell_probs = [0.5, 0.4, 0.1]
            cells[col, row] = random.choices(env_cell_types, cell_probs)[0]

def simple_grid(cells, start,finish, env, obstacle='grass'):
    
    for row in range(start,finish):
        obst_col=random.choices(range(env.width), [1/3.]*3)[0]
        cells[obst_col, row] = obstacle
        # 1 obstacle with prob 0.7, 2 obstacles with prob 0.2, 3 with prob 0.1
        extra_obst = random.choices(range(3), [0.7, 0.2, 0.1])[0]
        # 2 obstacles case, make sure the optimal path == same as greedy
        if extra_obst==1:
            # if row >0:
            #     obst_mask = [1 if cells[c, row-1]==obstacle else 0 for c in range(3)]
            #     if obst_mask[1]==1 and sum(obst_mask)==1:
            #         cells[0, row] = obstacle
            #         cells[1, row] = 'road'
            #         cells[2, row] = obstacle
            #         continue
            #     elif sum(obst_mask) == 1:
            #         if obst_col==1:
            #             second_obst_col = np.argwhere(np.array(obst_mask) == 1).flatten()[0]
            #             cells[second_obst_col, row] = obstacle
            #             cells[2 - second_obst_col, row] = 'road'
            #             continue
            #         elif obst_col==0 and obst_mask[2]==1:
            #             cells[2, row] = obstacle
            #             cells[1, row] = 'road'
            #             continue
            #         elif obst_col==2 and obst_mask[0]==1:
            #             cells[0, row] = obstacle
            #             cells[1, row] = 'road'
            #             continue
                    

              
            # choose remaining cell types
            second_obst_col = random.choices([int(obst_col==0), int(obst_col<=1) + 1], [0.5]*2)[0]
            cells[second_obst_col, row] = obstacle
            cells[3 - obst_col - second_obst_col, row] = 'road'

        else:
            for col in range(env.width):
                if col!=obst_col:
                    cells[col, row] = 'road' if not extra_obst else obstacle

            

                    