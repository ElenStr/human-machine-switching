# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from experiments.experiments import train, evaluate
from experiments.utils import *
from agent.agents import *
from agent.switching_agents import *
from environments.env import * 
from definitions import ROOT_DIR

from torch.optim import RMSprop
import numpy as np
import random 
import torch
import pickle
from copy import deepcopy


np.random.seed(12345678)
random.seed(12345678)
torch.manual_seed(12345678)

# %% [markdown]
# ## Train baselines and 2-stage algorithm
# %% [markdown]
# ### Configure Environment

# %%
env_generator = Environment()
width = 3
height = 20
depth = height//3
init_traffic_level = 'light'
env_params = {'width' : width, 'height':height, 'init_traffic_level': init_traffic_level, 'depth': depth}
env_generator_fn = lambda:env_generator.generate_grid_world(**env_params)
n_actions = 3

# %% [markdown]
# ### Configure Agents

# %%
# Human 
estimation_noise = 2.0
switching_noise = 0.0
c_H = 0.0

# Machine

# state size with string features
n_state_features_strings = env_generator.n_state_strings(depth, width)

# define state size in 1-hot encoding
n_state_features_1hot =  env_generator.n_state_one_hot(depth, width)

n_state_features = (n_state_features_strings, n_state_features_1hot)

c_M = 0.2
lr = 1e-4
optimizer_fn = lambda params: RMSprop(params, lr)

# Switching Agent
epsilon = 0.1

# %% [markdown]
# ### Initialize Agents

# %%
human  = NoisyDriverAgent(env_generator, noise_sd=estimation_noise, noise_sw=switching_noise, c_H=c_H)

machine = MachineDriverAgent(n_state_features, n_actions, optimizer_fn, c_M=c_M)

machine_only = FixedSwitchingMachine(n_state_features, optimizer_fn, c_M=c_M)

human_only = FixedSwitchingHuman()

switch_fixed_policies = SwitchingAgent(n_state_features, optimizer_fn, c_M=c_M, c_H=c_H, eps=epsilon)

# same initialisation
switch_full = SwitchingAgent(n_state_features, optimizer_fn, c_M=c_M, c_H=c_H, eps=epsilon)
# must be deepcopy of machine before training
switch_machine = MachineDriverAgent(n_state_features, n_actions, optimizer_fn, c_M=c_M)

# %% [markdown]
# ### Train Steps

# %%
n_traj = 50000
n_episodes = 25000

# %% [markdown]
# ### Gather human traces

# %%
# gather human trajectories
trajectories = gather_human_trajectories(human, env_generator, n_traj,**env_params)
# save for later
with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}', 'wb') as file:
    pickle.dump(trajectories, file, pickle.HIGHEST_PROTOCOL)
with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}_agent', 'wb') as file:
    pickle.dump(human, file, pickle.HIGHEST_PROTOCOL)    


# %%

# with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}', 'rb') as file:
#     trajectories = pickle.load(file)
# with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}_agent', 'rb') as file:
#     human = pickle.load(file)
# with open(f'{ROOT_DIR}/outputs/agents/pre_trained_machine/actor_agent_off', 'rb') as file:
#     machine = pickle.load(file)
# with open(f'{ROOT_DIR}/outputs/agents/pre_trained_machine/switching_agent_off', 'rb') as file:
#     machine_only = pickle.load(file)
# %% [markdown]
# ### Evaluation Parametes

# %%
n_eval_set_size = 1000
eval_freq = 1000
save_freq = 5000
eval_set = [ env_generator_fn() for i in range(n_eval_set_size)]

# %% [markdown]
# ### Training

# %%
machine_only_algo = {'pre_trained_machine_50K_fxH': (machine_only, [human, machine])}
machine_only_algo, machine_only_costs = train(machine_only_algo, trajectories, env_generator_fn, n_episodes, eval_set, eval_freq, save_freq)


# %%



machine.trainable = False

algos = {'fixed_policies_50K_fxH': (switch_fixed_policies,[human, machine]), 'switching_50K_fxH':( switch_full,[human, switch_machine]) }
algos, algos_costs = train(algos, trajectories, env_generator_fn, n_episodes, eval_set, eval_freq, save_freq)
