from  numpy import sqrt
# from environments.env import *
from environments.taxi_env import MapEnv 
from environments.utils_env import *
import osmnx as ox
from data.preprocess_data import add_trip_ids_to_nodes
from definitions import PYTHON_VERSION
if PYTHON_VERSION < 9:
    import pickle5 as pickle
else:
    import pickle

# Load data
final_graph_path = 'data/final_graph.osm'
trips_dict_path = 'data/trips.pkl'
final_trips_path = 'data/cleaned_up_trips.csv'
print("Loading trips")
with open(trips_dict_path, 'rb') as f:
    TRIPS = pickle.load(f)
data_size = len(TRIPS)
# Environment 
print("Loading graph")
graph = ox.graph_from_xml(final_graph_path, simplify=False)
print("Adding trips to nodes")
gr = add_trip_ids_to_nodes(final_trips_path, graph)
ENV = MapEnv(gr, TRIPS)
n_actions = ENV.MAX_OUT_DEGREE


# TODO: run with different random splits for train/test set 
# Configure runs with different seeds

seed_list = [87651234, 12367845, 12783645, 18453627, 37468124, 38294734,2938472,49264719, 58375625]
# exepetiment will run (run_end - run_start) times
run_start = 0
run_end = 1

# Setting and agent config
# No nned for setting now
# setting = 2 # setting is the same as 'scenario' in paper. set 2 for I, 3 for II and 7 for III  
agent = f'auto'# agent can be {auto, fxd, switch} == {machine, fixSwitch, triage}
# agent+=f'V{setting}{scen_postfix}' 
method = 'off'
# actual_human = True
entropy_weight = 0.01

# Fraction of trips for off and online training
offline_train_split = 0.1 # Train split for offline training
online_train_split = 0.7 - offline_train_split  # Train split for online training
n_try = 1 # Recorded trips per source destination pair


# Human 
c_H = 0

# Machine
batch_size = 1
# if setting == 2:
#     obstacle_to_ignore = 'grass'
# elif setting == 7:
#     obstacle_to_ignore = 'stone'  
# else:
#     obstacle_to_ignore = ''  
c_M = 0
lr = 1e-4

# Switching Agent
# epsilon schedule
def eps_fn(timestep):
    # Try simplest schedule for now 
    epsilon = 0.1

    # off_steps = n_traj*n_try if 'off' in method else 0
    # if timestep < off_steps*19//2 :
    #     epsilon = 0.2
    # elif timestep < off_steps*19:
    #     epsilon = 0.1
    # else:
    #     scaled_time = (timestep - off_steps*19+1)//19000 + 1
    #     epsilon = 0.1* 1 / sqrt(scaled_time)
    return epsilon

epsilon = eps_fn 



# Saving and evaluation
eval_split = 1 - offline_train_split - online_train_split # Test set size
eval_freq = 5000
save_freq = 25000//batch_size
eval_tries = 1 #number of evaluation runs



# Scenarios stand for different enviornment types (not the scenarios in paper)

# other_snenarios = [lambda c,s,f: fn(c,s,f,obst) for fn in [two_lanes_obstcales, difficult_grid] for obst in ['car', 'grass']]
# scenarios = [lambda c,s,f: general_grid(c,s,f,env_generator),lambda c,s,f: general_grid(c,s,f,env_generator) ] #+other_snenarios  
# # scenarios = [lambda c,s,f: clean_grass(c,s,f,env_generator), lambda c,s,f: clean_car(c,s,f,env_generator)  ] #+other_snenarios  
# # scenarios = [lambda c,s,f: simple_grid(c,s,f,env_generator, 'grass'), lambda c,s,f: simple_grid(c,s,f,env_generator, 'car')  ] #+other_snenarios  

# scen_postfix = '_scGen' if len(scenarios) > 1 else ''
# def env_generator_fn(n_grids):
#     grids = []
#     n_grids_per_scenario = n_grids #// len(scenarios)
#     all_env_params = {'scenario_fn': scenarios[1], 'base_fn': scenarios[0], **env_params}
#     grids_sc = [env_generator.generate_grid_world(**all_env_params) for _ in range(n_grids_per_scenario)]
#     grids.extend(grids_sc)
#     random.shuffle(grids)
#     return grids



