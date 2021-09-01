import networkx
import numpy as np

import osmnx as ox
# state: (current node number, destination node number)


def get_lat_lon(G, node_number):
    node_dict = G.nodes[node_number]
    return [node_dict['y'], node_dict['x']]


class MapEnv:
    def __init__(self, graph: networkx.classes.multidigraph.MultiDiGraph):
        self.G = graph
        self.current_node = None
        self.dest_node = None

    def reset(self, state: tuple):
        """
        state: (current node number, destination node number)
        """
        self.current_node = state[0]
        self.dest_node = state[1]

    def step(self, action: int):
        """
        returns: state, cost, finished
        """
        edge = list(self.G.out_edges(self.current_node))[action]
        # TODO: handle dead-end nodes!!
        # if len(edges) < 1:
        #     edges = list(self.G.in_edges(self.current_node))

        self.current_node = edge[1]
        cost = self._find_edge_cost(edge)

        return self.current_state(), cost, self.current_node == self.dest_node

    def current_state(self):
        return self.current_node, self.dest_node

    def _find_edge_cost(self, edge: tuple):
        cc = ox.distance.great_circle_vec(*get_lat_lon(self.G, edge[1]), *get_lat_lon(self.G, self.dest_node))
        return self.G.get_edge_data(edge[0], edge[1])[0]['length'] + cc


class FeatureHandler:
    def __init__(self, graph: networkx.classes.multidigraph.MultiDiGraph):
        self.G = graph
        self.max_actions = max([self.G.out_degree(n) for n in self.G.nodes])
        self.feature_size = (self.max_actions + 2) * 2

    def state2feature(self, state: tuple):
        """
        state: (current node number, destination node number)
        """
        # current node, destination node, output nodes
        all_nodes = [state[0]] + [state[1]] + [e[1] for e in self.G.out_edges(state[0])]
        features = np.array([get_lat_lon(self.G, n) for n in all_nodes]).flatten()
        return np.pad(features, (0, self.feature_size - len(features)), mode='constant')

    def action_numbers(self, state: tuple):
        """
        state: (current node number, destination node number)
        """
        return self.G.out_degree(state[0])
