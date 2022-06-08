from experiments.experiments import train, evaluate
from experiments.utils import *
from agent.agents import *
from agent.switching_agents import *
# from environments.env import * 
from definitions import ROOT_DIR, PYTHON_VERSION
from config import *
from sklearn.model_selection import train_test_split

from torch.optim import RMSprop,Adam
import os
import numpy as np
import random 
import torch
if PYTHON_VERSION < 9:
    import pickle5 as pickle
else:
    import pickle
import sys
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)
for run in range(run_start,run_end):
    
    cur_seed = seed_list[run]
    np.random.seed(cur_seed)
    random.seed(cur_seed)
    torch.manual_seed(cur_seed)



    dir_post_fix = ''
    # traj_post_fx = '_pureState'

    # trajectories = []
    # on_line_set = []
    human = None

    out_dir = f'{ROOT_DIR}/outputs'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    res_dir = f'{ROOT_DIR}/results'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)  
    
    print(f"Created results paths")

   # split ids for offline and online training and testing
    trips_train, test_trips = train_test_split(list(TRIPS.keys()), test_size=eval_split, random_state=42)
    offline_trips, online_trips = train_test_split(trips_train, test_size=online_train_split,random_state=42)

    print(f"Splits Done, Offline train size :{len(offline_trips)}, offline sample:{offline_trips[:10]}")

    dir_name = f"{agent}_b{batch_size}_{'W' if entropy_weight > 0. else 'N'}e{dir_post_fix}_{'' if eval_tries == 1 else f'e{eval_tries}_'}"
    dir_name+=f'_run{run}'
    res_path = f'{ROOT_DIR}/results/{dir_name}'
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    print(f"Experiment results path created")

    # Human agent
    human = HumanTaxiAgent(ENV, c_H)
    print(f"Human agent created")
    #  state size get coords in (angle, dist) for current-ref, current-dest and each neighbor-dest
    n_state_features = 4 + ENV.MAX_OUT_DEGREE*2
    optimizer_fn = lambda params: Adam(params, lr)
    machine = MachineDriverAgent(n_state_features, n_actions, optimizer_fn, c_M=c_M, entropy_weight=entropy_weight, batch_size=batch_size)

    print(f"Machine agent created")

    if 'auto' in agent:
        switch_agent = FixedSwitchingMachine(n_state_features, optimizer_fn, c_M=c_M, batch_size=batch_size)
    else:
        switch_agent = SwitchingAgent(n_state_features, optimizer_fn, c_M=c_M, c_H=c_H, eps_fn=epsilon, batch_size=batch_size)
    print(f"Switching agent created")

    if 'fxd' in agent:
        # TODO make it work for any method of auto, now works only for same auto and fxd methods
        start_rest = 1 
        machine_agent_name = f'auto_'+'_'.join(list(filter(lambda x: x!='e3', dir_name.split('_')[start_rest:])))
        machine_dir = f'{ROOT_DIR}/results/{machine_agent_name}/actor_agent_off'
        
        try:
            with open(machine_dir, 'rb') as file:
                machine = pickle.load(file) 
                
        except:
            
            if not os.path.exists(machine_dir):
                os.mkdir( f'{ROOT_DIR}/results/{machine_agent_name}')
            machine_only = FixedSwitchingMachine(n_state_features, optimizer_fn, c_M=c_M, batch_size=batch_size)
            machine_algo = {machine_agent_name: (machine_only, [human, machine])}
        
            machine_algo, costs = train(machine_algo, offline_trips,[], test_trips, eval_freq,  save_freq, batch_size=batch_size, eval_tries=1)
            machine = machine_algo[machine_agent_name][1][1]
        
        machine.trainable = False
    

    
    algo = {dir_name: (switch_agent, [human, machine])}
    orig_stdout = sys.stdout
    orig_err = sys.stderr

    print(len(offline_trips))
    with open(f'{ROOT_DIR}/{dir_name}_err.out','w', buffering=1) as ferr:
        with open(f'{ROOT_DIR}/{dir_name}.out','w', buffering=1) as f:
            sys.stdout = f
            sys.stderr = ferr
            try:
                algo, costs = train(algo, offline_trips, online_trips, test_trips, eval_freq, save_freq, batch_size=batch_size, eval_tries=eval_tries)
                sys.stdout = orig_stdout
            except Exception as e:
                
                sys.stdout = orig_stdout
                sys.stderr = orig_err   
                print(e)     
                
        sys.stderr = orig_err

