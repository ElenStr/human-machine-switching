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

# Scenarios stand for different enviornment types (not the scenarios in paper)

# other_snenarios = [lambda c,s,f: fn(c,s,f,obst) for fn in [two_lanes_obstcales, difficult_grid] for obst in ['car', 'grass']]
scenarios = [lambda c,s,f: general_grid(c,s,f,env_generator),lambda c,s,f: general_grid(c,s,f,env_generator) ] #+other_snenarios  
# scenarios = [lambda c,s,f: clean_grass(c,s,f,env_generator), lambda c,s,f: clean_car(c,s,f,env_generator)  ] #+other_snenarios  
# scenarios = [lambda c,s,f: simple_grid(c,s,f,env_generator, 'grass'), lambda c,s,f: simple_grid(c,s,f,env_generator, 'car')  ] #+other_snenarios  

scen_postfix = '_scGen' if len(scenarios) > 1 else ''
def env_generator_fn(n_grids):
    grids = []
    n_grids_per_scenario = n_grids #// len(scenarios)
    all_env_params = {'scenario_fn': scenarios[1], 'base_fn': scenarios[0], **env_params}
    grids_sc = [env_generator.generate_grid_world(**all_env_params) for _ in range(n_grids_per_scenario)]
    grids.extend(grids_sc)
    random.shuffle(grids)
    return grids

# Setting and agent config
setting = 2 # setting is the same as 'scenario' in paper. set 2 for I, 3 for II and 7 for III  
agent = f'switch'# agent can be {auto, fxd, switch} == {machine, fixSwitch, triage}
agent+=f'V{setting}{scen_postfix}' 
method = 'off_on'
actual_human = True
entropy_weight = 0.01

# Number of episodes for off and online training
n_traj = 60000 # number of grids in which human acts alone
n_try = 1 # number of human trajectories per grid to be recorded
n_episodes = 100000 # online training episodes

# Human 
estimation_noise = 0.0 #probablity picking at random
p_ignore = 1.0 
c_H = 1.0 if setting==7 else 0

# Machine
batch_size = 1
if setting == 2:
    obstacle_to_ignore = 'grass'
elif setting == 7:
    obstacle_to_ignore = 'stone'  
else:
    obstacle_to_ignore = ''  
c_M = 1 if setting==3 else 0.0
lr = 1e-4

# Switching Agent
# epsilon schedule
def eps_fn(timestep):
    off_steps = n_traj*n_try if 'off' in method else 0
    if timestep < off_steps*19//2 :
        epsilon = 0.2
    elif timestep < off_steps*19:
        epsilon = 0.1
    else:
        scaled_time = (timestep - off_steps*19+1)//19000 + 1
        epsilon = 0.1* 1 / sqrt(scaled_time)
    return epsilon

epsilon = eps_fn 



# Saving and evaluation
n_eval = 1000
eval_freq = 1000
save_freq = 5000//batch_size
eval_tries = 1 #number of evaluation runs