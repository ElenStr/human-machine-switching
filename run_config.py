from experiments.experiments import train, evaluate
from experiments.utils import *
from agent.agents import *
from agent.switching_agents import *
from environments.env import * 
from definitions import ROOT_DIR
from config import *

from torch.optim import RMSprop
import os
import numpy as np
import random 
import torch
import pickle
import sys
from copy import deepcopy

for run in range(run_start,run_end):
    
    cur_seed = seed_list[run]
    np.random.seed(cur_seed)
    random.seed(cur_seed)
    torch.manual_seed(cur_seed)


    dir_post_fix = ''
    traj_post_fx = '_pureState'

    trajectories = []
    on_line_set = []
    human = None

    out_dir = f'{ROOT_DIR}/outputs'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    res_dir = f'{ROOT_DIR}/results'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)  

    # gather trajecotries for offline training
    if 'off' in method:
        traj_path = f'{ROOT_DIR}/outputs/trajectories'
        human_path = f'/human{setting}_{estimation_noise}_0.0_{init_traffic_level}_trajectories_{n_traj}_{n_try}'
        dir_post_fix = f'_off{"T" if actual_human else "F"}_D{n_traj / 1000}K{n_try}R'
        try:
            with open(traj_path+human_path+traj_post_fx+scen_postfix, 'rb') as file:
                trajectories = pickle.load(file)
            with open(traj_path+human_path+'_agent'+traj_post_fx+scen_postfix, 'rb') as file:
                human = pickle.load(file)
        except:
            if not os.path.exists(traj_path):
                os.mkdir(traj_path)
            human  = NoisyDriverAgent(env_generator, prob_wrong=estimation_noise, p_ignore_car= p_ignore ,setting=setting, c_H=c_H)
            trajectories = []
            n_grids_per_scenario = n_traj #// len(scenarios)
            
            # for scen_fn in scenarios:
            all_env_params = {'scenario_fn': scenarios[1],'base_fn': scenarios[0], **env_params}
            traj_sc = gather_human_trajectories(human, env_generator,n_grids_per_scenario,n_try ,**all_env_params) 
            trajectories.extend(traj_sc)
            random.shuffle(trajectories)
            with open(traj_path+human_path+traj_post_fx+scen_postfix, 'wb') as file:
                pickle.dump(trajectories, file, pickle.HIGHEST_PROTOCOL)
            with open(traj_path+human_path+'_agent'+traj_post_fx+scen_postfix, 'wb') as file:
                pickle.dump(human, file, pickle.HIGHEST_PROTOCOL)

    # generate episodes for online training
    if 'on' in method :
        ds_on_path = f'{ROOT_DIR}/outputs/on_line_set_{n_episodes}_{init_traffic_level}{scen_postfix}'
        dir_post_fix += f'_on_D{n_episodes/1000}K'
        if human is None:
            human = NoisyDriverAgent(env_generator, prob_wrong=estimation_noise,p_ignore_car= p_ignore, setting=setting, c_H=c_H)

        try:
            with open(ds_on_path, 'rb') as file:
                on_line_set = pickle.load(file)
        except:
            on_line_set = env_generator_fn(n_episodes)
            with open(ds_on_path, 'wb') as file:
                pickle.dump(on_line_set, file, pickle.HIGHEST_PROTOCOL)
        
            
    # generate episodes for testing
    try:
        eval_path = f'{ROOT_DIR}/outputs/eval_set{scen_postfix}'
        with open(eval_path, 'rb') as file:
            eval_set = pickle.load(file)
    except:
        eval_set = env_generator_fn(n_eval)
        with open(eval_path, 'wb') as file:
            pickle.dump(eval_set, file, pickle.HIGHEST_PROTOCOL)



    dir_name = f"{agent}_b{batch_size}_{'W' if entropy_weight > 0. else 'N'}e{dir_post_fix}_{'' if eval_tries == 1 else f'e{eval_tries}_'}h{estimation_noise}"
    dir_name+=f'_run{run}'
    res_path = f'{ROOT_DIR}/results/{dir_name}'
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    n_state_features_strings = env_generator.n_state_strings(depth, width)
    # define state size in 1-hot encoding
    n_state_features_1hot =  env_generator.n_state_one_hot(depth, width)
    n_state_features = (n_state_features_strings, n_state_features_1hot)
    optimizer_fn = lambda params: RMSprop(params, lr)
    machine = MachineDriverAgent(n_state_features, n_actions, optimizer_fn, c_M=c_M, entropy_weight=entropy_weight, setting=setting, batch_size=batch_size)

    if 'auto' in agent:
        switch_agent = FixedSwitchingMachine(n_state_features, optimizer_fn, c_M=c_M, batch_size=batch_size)
    else:
        switch_agent = SwitchingAgent(n_state_features, optimizer_fn, c_M=c_M, c_H=c_H, eps_fn=epsilon, batch_size=batch_size)

    if 'fxd' in agent:
        # TODO make it work for any method of auto, now works only for same auto and fxd methods
        start_rest = 1 if scen_postfix == ''else 2
        machine_agent_name = f'auto{setting}{scen_postfix}_'+'_'.join(list(filter(lambda x: x!='e3', dir_name.split('_')[start_rest:])))
        machine_agent_name+=f'_run{run}'
        machine_dir = f'{ROOT_DIR}/results/{machine_agent_name}/actor_agent_off'
        
        try:
            with open(machine_dir, 'rb') as file:
                machine = pickle.load(file) 
                
        except:
            
            if not os.path.exists(machine_dir):
                os.mkdir( f'{ROOT_DIR}/results/{machine_agent_name}')
            machine_only = FixedSwitchingMachine(n_state_features, optimizer_fn, c_M=c_M, batch_size=batch_size)
            machine_algo = {machine_agent_name: (machine_only, [human, machine])}
        
            machine_algo, costs = train(machine_algo, trajectories,[], eval_set, eval_freq,  save_freq, batch_size=batch_size, eval_tries=1)
            machine = machine_algo[machine_agent_name][1][1]
        
        machine.trainable = False
    

    human.actual = actual_human
    algo = {dir_name: (switch_agent, [human, machine])}
    orig_stdout = sys.stdout
    orig_err = sys.stderr
    with open(f'{ROOT_DIR}/{dir_name}_err.out','w', buffering=1) as ferr:
        with open(f'{ROOT_DIR}/{dir_name}.out','w', buffering=1) as f:
            sys.stdout = f
            sys.stderr = ferr
            try:
                algo, costs = train(algo, trajectories, on_line_set, eval_set, eval_freq, save_freq, batch_size=batch_size, eval_tries=eval_tries)
                sys.stdout = orig_stdout
            except Exception as e:

                sys.stdout = orig_stdout
                sys.stderr = orig_err        
                
        sys.stderr = orig_err

