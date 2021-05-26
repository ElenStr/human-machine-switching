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
batch_size = 5
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

machine = MachineDriverAgent(n_state_features, n_actions, optimizer_fn, c_M=c_M, entropy_weight=0.01, batch_size=batch_size)

machine_only = FixedSwitchingMachine(n_state_features, optimizer_fn, c_M=c_M, batch_size=batch_size)

human_only = FixedSwitchingHuman()

switch_fixed_policies = SwitchingAgent(n_state_features, optimizer_fn, c_M=c_M, c_H=c_H, eps=epsilon, batch_size=batch_size)

# same initialisation
switch_full = SwitchingAgent(n_state_features, optimizer_fn, c_M=c_M, c_H=c_H, eps=epsilon, batch_size=batch_size)
# must be deepcopy of machine before training
switch_machine = MachineDriverAgent(n_state_features, n_actions, optimizer_fn, c_M=c_M, entropy_weight=0.01, batch_size=batch_size)

# %% [markdown]
# ### Train Steps

# %%
n_traj = 50000
n_episodes = 50000

# %% [markdown]
# ### Gather human traces
human = RandomDriverAgent()
trajectories = gather_human_trajectories(human, env_generator, n_traj,**env_params)
# # save for later
with open(f'{ROOT_DIR}/outputs/trajectories/random_human_{init_traffic_level}_trajectories_{n_traj}', 'wb') as file:
    pickle.dump(trajectories, file, pickle.HIGHEST_PROTOCOL)
with open(f'{ROOT_DIR}/outputs/trajectories/random_human_{init_traffic_level}_trajectories_{n_traj}_agent', 'wb') as file:
    pickle.dump(human, file, pickle.HIGHEST_PROTOCOL) 
# %%
# gather human trajectories
# trajectories = gather_human_trajectories(human, env_generator, n_traj,**env_params)
# # save for later
# with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}', 'wb') as file:
#     pickle.dump(trajectories, file, pickle.HIGHEST_PROTOCOL)
# with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}_agent', 'wb') as file:
#     pickle.dump(human, file, pickle.HIGHEST_PROTOCOL)    

# on_line_set = [env_generator_fn() for i in range(n_episodes)]
# with open(f'{ROOT_DIR}/outputs/on_line_set_{n_episodes}_{init_traffic_level}', 'wb') as file:
#     pickle.dump(on_line_set, file, pickle.HIGHEST_PROTOCOL)

# %%

with open(f'{ROOT_DIR}/outputs/on_line_set_{n_episodes}_{init_traffic_level}', 'rb') as file:
    on_line_set = pickle.load(file)

# with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}', 'rb') as file:
#     trajectories = pickle.load(file)
# with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_trajectories_{n_traj}_agent', 'rb') as file:
#     human = pickle.load(file)
# with open(f'{ROOT_DIR}/outputs/agents/machine_only_50K_2L/actor_agent_off', 'rb') as file:
#     machine = pickle.load(file)
# with open(f'{ROOT_DIR}/outputs/agents/machine_only_50K_2L/switching_agent_off', 'rb') as file:
#     machine_only = pickle.load(file)
# %% [markdown]
# ### Evaluation Parametes
# %%


n_eval_set_size = 1000
eval_freq = 1000//batch_size
save_freq = 5000//batch_size
# # eval_set = [ env_generator_fn() for i in range(n_eval_set_size)]
# # with open(f'{ROOT_DIR}/outputs/eval_set', 'wb')as file:
# #     pickle.dump(eval_set, file, pickle.HIGHEST_PROTOCOL)
with open(f'{ROOT_DIR}/outputs/eval_set', 'rb') as file:
    eval_set = pickle.load(file)

# human_cost = evaluate(human_only, [human], eval_set)
# with open(f'{ROOT_DIR}/outputs/trajectories/random_human_{estimation_noise}_{switching_noise}_{init_traffic_level}_cost', 'wb') as file :
#     pickle.dump(human_cost, file, pickle.HIGHEST_PROTOCOL)
# human_cost = evaluate(human_only, [human], eval_set)
# with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_cost', 'wb') as file :
#     pickle.dump(human_cost, file, pickle.HIGHEST_PROTOCOL)
with open(f'{ROOT_DIR}/outputs/trajectories/human_{estimation_noise}_{switching_noise}_{init_traffic_level}_cost', 'rb') as file :
    human_cost = pickle.load (file)
print(f'Human cost {human_cost}')
# # %% [markdown]
# # ### Training
# for i in range(50000):
#     on_line_set.append(env_generator_fn())
# with open(f'{ROOT_DIR}/outputs/on_line_set_100000_{init_traffic_level}', 'wb') as file:
#     pickle.dump(on_line_set, file, pickle.HIGHEST_PROTOCOL)
# # %%

machine_only_algo = {'machine_off_rnd_human': (machine_only, [human, machine])}
machine_only_algo, machine_only_costs = train(machine_only_algo, trajectories, [], eval_set, eval_freq, save_freq, batch_size=batch_size)

# machine.trainable = False
# algos = {'fixed_policies_50K': (switch_fixed_policies,[human, machine])}
# algos, algos_costs = train(algos, trajectories, [], eval_set, eval_freq, save_freq, batch_size=batch_size,eval_tries=5)


# machine_only_algo = {'machine_only_50K-50K': (machine_only, [human, machine])}
# machine_only_algo, machine_only_costs = train(machine_only_algo, [], on_line_set, eval_set, eval_freq, save_freq, not_batched=False)
# # # # %%

# algos =  { 'switching_50K':( switch_full,[human, switch_machine]) }
# algos, algos_costs = train(algos, trajectories, [], eval_set, eval_freq, save_freq, batch_size=batch_size,eval_tries=5)


# algos =  { 'switching_50K-50K':( switch_full,[human, switch_machine]) }
# algos, algos_costs = train(algos, [], on_line_set, eval_set, eval_freq, save_freq, not_batched=False,eval_tries=5)
