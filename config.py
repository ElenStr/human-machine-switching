from  numpy import sqrt
from environments.env import * 
from environments.utils_env import *
# Environment 
width = 3
height = 20
depth = height//3
init_traffic_level = 'light'
n_actions = 3

env_generator = Environment()
env_params = { 'width' : width, 'height':height, 'init_traffic_level': init_traffic_level, 'depth': depth}

other_snenarios = [lambda c,s,f: fn(c,s,f,obst) for fn in [two_lanes_obstcales, difficult_grid] for obst in ['car', 'grass']]
# scenarios = [lambda c,s,f: general_grid(c,s,f,env_generator) ] #+other_snenarios  
scenarios = [lambda c,s,f: clean_grass(c,s,f,env_generator), lambda c,s,f: clean_car(c,s,f,env_generator)  ] #+other_snenarios  

scen_postfix = '_sc' if len(scenarios) > 1 else ''
def env_generator_fn(n_grids):
    grids = []
    n_grids_per_scenario = n_grids // len(scenarios)
    for scen_fn in scenarios:
        all_env_params = {'scenario_fn': scen_fn, 'base_fn': scen_fn, **env_params}
        grids_sc = [env_generator.generate_grid_world(**all_env_params) for _ in range(n_grids_per_scenario)]
        grids.extend(grids_sc)
    random.shuffle(grids)
    return grids
# Setting and agent config
setting = 2
agent = f'switch{setting}{scen_postfix}'
method = 'off_on'
actual_human = True
entropy_weight = 0.01

# Dataset sizes for off and online training
n_traj = 100000
n_try = 1
n_episodes = 100000

# Human 
estimation_noise = 0.2
switching_noise = 0.0
p_ignore = 1.0
c_H = 0.0

# Machine
batch_size = 1
c_M = 0.2 if setting==1 else c_H
lr = 1e-4

# Switching Agent
# epsilon schedule
def eps_fn(timestep):
    off_steps = n_traj*n_try if 'off' in method else 0
    if timestep < off_steps*20//2 :
        epsilon = 0.2
    elif timestep < off_steps*20:
        epsilon = 0.1
    else:
        scaled_time = (timestep - off_steps*20+1)//20000 + 1
        epsilon = 0.1* 1 / sqrt(scaled_time)
    return epsilon

epsilon = eps_fn 



# Saving and evaluation
n_eval = 1000
eval_freq = 1000
save_freq = 5000//batch_size
eval_tries = 1 if 'auto' in agent else 3