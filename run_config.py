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
from copy import deepcopy


np.random.seed(12345678)
random.seed(12345678)
torch.manual_seed(12345678)

env_generator = Environment()
env_params = {'width' : width, 'height':height, 'init_traffic_level': init_traffic_level, 'depth': depth}
env_generator_fn = lambda:env_generator.generate_grid_world(**env_params)
dir_post_fix = ''
trajectories = []
on_line_set = []
human = None

out_dir = f'{ROOT_DIR}/outputs'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
res_dir = f'{ROOT_DIR}/results'
if not os.path.exists(res_dir):
    os.mkdir(res_dir)  

if 'off' in method:
    traj_path = f'{ROOT_DIR}/outputs/trajectories'
    human_path = f'/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}'
    dir_post_fix = f'_off_D{str(n_traj)[:-3]}K'
    try:
        with open(traj_path+human_path, 'rb') as file:
            trajectories = pickle.load(file)
        with open(traj_path+human_path+'_agent', 'rb') as file:
            human = pickle.load(file)
    except:
        if not os.path.exists(traj_path):
            os.mkdir(traj_path)
        human  = NoisyDriverAgent(env_generator, noise_sd=estimation_noise, noise_sw=switching_noise, c_H=c_H)
        trajectories = gather_human_trajectories(human, env_generator, n_traj,**env_params)
        
        with open(traj_path+human_path, 'wb') as file:
            pickle.dump(trajectories, file, pickle.HIGHEST_PROTOCOL)
        with open(traj_path+human_path+'_agent', 'wb') as file:
            pickle.dump(human, file, pickle.HIGHEST_PROTOCOL)

if 'on' in method :
    ds_on_path = f'{ROOT_DIR}/outputs/on_line_set_{n_episodes}_{init_traffic_level}'
    dir_post_fix += f'_on_D{str(n_episodes)[:-3]}K'
    if human is None:
        human = NoisyDriverAgent(env_generator, noise_sd=estimation_noise, noise_sw=switching_noise, c_H=c_H)

    try:
        with open(ds_on_path, 'rb') as file:
            on_line_set = pickle.load(file)
    except:
        on_line_set = [env_generator_fn() for i in range(n_episodes)]
        with open(ds_on_path, 'wb') as file:
            pickle.dump(on_line_set, file, pickle.HIGHEST_PROTOCOL)

try:
    eval_path = f'{ROOT_DIR}/outputs/eval_set'
    with open(eval_path, 'rb') as file:
        eval_set = pickle.load(file)
except:
    eval_set = [env_generator_fn() for i in range(n_eval)]
    with open(eval_path, 'wb') as file:
        pickle.dump(eval_set, file, pickle.HIGHEST_PROTOCOL)



dir_name = f"{agent}_b{batch_size}_{'W' if entropy_weight > 0. else 'N'}e{dir_post_fix}_{'' if eval_tries == 1 else f'e{eval_tries}_'}h{estimation_noise}"

res_path = f'{ROOT_DIR}/results/{dir_name}'
if not os.path.exists(res_path):
    os.mkdir(res_path)

n_state_features_strings = env_generator.n_state_strings(depth, width)
# define state size in 1-hot encoding
n_state_features_1hot =  env_generator.n_state_one_hot(depth, width)
n_state_features = (n_state_features_strings, n_state_features_1hot)
optimizer_fn = lambda params: RMSprop(params, lr)
machine = MachineDriverAgent(n_state_features, n_actions, optimizer_fn, c_M=c_M, entropy_weight=entropy_weight, batch_size=batch_size)

if 'auto' in agent:
    switch_agent = FixedSwitchingMachine(n_state_features, optimizer_fn, c_M=c_M, batch_size=batch_size)
else:
    switch_agent = SwitchingAgent(n_state_features, optimizer_fn, c_M=c_M, c_H=c_H, eps_fn=epsilon, batch_size=batch_size)

if 'fxd' in agent:
    # TODO make it work for any method of auto, now works only for same auto and fxd methods
    machine_agent_name = 'auto_'+'_'.join(list(filter(lambda x: x!='e3', dir_name.split('_')[1:])))
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


algo = {dir_name: (switch_agent, [human, machine])}
algo, costs = train(algo, trajectories, on_line_set, eval_set, eval_freq, save_freq, batch_size=batch_size, eval_tries=eval_tries)
