from  numpy import sqrt

# Environment 
width = 3
height = 20
depth = height//3
init_traffic_level = 'light'
n_actions = 3

# Human 
estimation_noise = 3.5
switching_noise = 0.0
c_H = 0.0

# Machine
batch_size = 1
c_M = 0.2
lr = 1e-4

# Switching Agent
# epsilon schedule
def eps_fn(timestep):
    if (timestep//20) < 25000:
        epsilon = 0.5
    elif (timestep//20) < 50000:
        epsilon = 0.1
    else:
        epsilon = 0.1* 1 / sqrt(timestep - 50000*20+1)
    return epsilon

epsilon = eps_fn 

# Dataset sizes for off and online training
n_traj = 50000
n_episodes = 50000

n_eval = 1000
eval_freq = 1000
save_freq = 5000//batch_size
eval_tries = 1

agent = 'auto1'
method = 'off_on'
entropy_weight = 0.01