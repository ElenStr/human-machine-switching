from environments.taxi_env import FeatureHandler, MapEnv
from agent.taxi_switching_agents import FixedSwitchingMachine
from experiments.taxi_utils import taxi_train_online
from agent.taxi_agents import MachineTaxiDriver

from torch.optim import RMSprop, Adam
import os
import numpy as np
import random
import torch
import pickle
import sys

import osmnx as ox
from copy import deepcopy


def handle_deadends(graph):
    edges = []
    for n in graph.nodes:
        if graph.out_degree(n) == 0:
            m = list(graph.in_edges(n))[0][0]
            edges.append((n, m, {'length': graph.get_edge_data(m, n)[0]['length']}))
    graph.add_edges_from(edges)
    return graph

np.random.seed(12345678)
random.seed(12345678)
torch.manual_seed(12345678)

# first: online fully automated machine algorithm
G = ox.graph_from_xml('./map-processing/data/Porto_driving.osm')
G = handle_deadends(G)

eval_n = 1
train_n = 10000
c_M = 0
lr = 1e-4
batch_size = 1
optimizer_critic_fn = lambda params: Adam(params, lr=1e-5)
optimizer_actor_fn = lambda params: Adam(params, lr)

# TODO: first only one destination
dest = 26016441
start = 1837688580
# online_eval_set = list(zip(random.choices(list(G.nodes), k=eval_n), [dest for i in range(eval_n)]))
online_eval_set = [(start, dest)]
train_set = []
for i in range(int(train_n / eval_n)):
    train_set.extend(online_eval_set)

feature_handler = FeatureHandler(G)

switch_agent = FixedSwitchingMachine(feature_handler, optimizer_critic_fn, c_M=c_M, batch_size=batch_size)

machine_agent = MachineTaxiDriver(feature_handler, optimizer_actor_fn, c_M)
# machine_agent.trainable = False
algo = {'online': (switch_agent, [None, machine_agent])}

env = MapEnv(G)
algo, costs = taxi_train_online(algo, train_set, online_eval_set, env, feature_handler, eval_freq=500, max_ep_l=100)