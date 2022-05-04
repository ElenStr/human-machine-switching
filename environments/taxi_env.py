import networkx
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians, sin, cos, atan2,degrees
import osmnx as ox
# state: (current node number, destination node number)


def get_lat_lon(G, node_number):
    node_dict = G.nodes[node_number]
    # 'y' is lat, 'x' is lon
    return [node_dict['y'], node_dict['x']]

def get_distance(G, node_id_u, node_id_v):
    """Haversine distance between two nodes"""
    
    u_coords =  get_lat_lon(G, node_id_u)
    v_coords =  get_lat_lon(G, node_id_v)

    u_coords_rad = list(map(radians, u_coords))
    v_coords_rad = list(map(radians, v_coords))

    distance_km = 6371 * haversine_distances([u_coords_rad, v_coords_rad])[0,1]
    return distance_km

def get_angle(G, node_id_u, node_id_v):
    """Angle between two nodes"""
    def angleFromCoordinate(point1,point2):
        lat1, long1 = point1
        lat2, long2 = point2

        dLon = (long2 - long1)

        y = sin(dLon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)

        brng = atan2(y, x)

        brng = degrees(brng)
        brng = (brng + 360) % 360
        brng = 360 - brng # count degrees clockwise - remove to make counter-clockwise

        return brng

    u_coords =  get_lat_lon(G, node_id_u)
    v_coords =  get_lat_lon(G, node_id_v)

    u_coords_rad = list(map(radians, u_coords))
    v_coords_rad = list(map(radians, v_coords))

    return angleFromCoordinate(u_coords_rad, v_coords_rad)







class MapEnv:
    def __init__(self, graph: networkx.classes.multidigraph.MultiDiGraph):
        self.G = graph
        self.MAX_OUT_DEGREE = max(list(map(lambda x: len(self.G.out_edges(x)), self.G.nodes)))
        #  Set reference node to distinuish between nodes
        self.reference_node = 25632226

        self.current_node = None
        self.distance_from_reference = None
        self.angle_from_reference = None
        self.dest_node = None
        self.neighbors = []
        self.neighbors_sorted = []

        
    def _set_curr_node_state_info(self, node_id):
        self.current_node = node_id
        self.distance_from_reference = get_distance(self.G, node_id, self.reference_node)
        self.angle_from_reference = get_angle(self.G, node_id, self.reference_node)
        # TODO: check what is the fastest subsciptable structure to use
        self.neighbors = list(self.G.neighbors(node_id))

    def reset(self, id_start, id_end):
        """
        Reset the info to retrieve the state to the beginning of new trip 
        """
        self.dest_node = id_end
        self._set_curr_node_state_info(id_start)
        self._set_angle_distance_neighbors_sorted()
        
       

    def step(self, action: int):
        """
        returns: state, cost, finished
        """
        # edge = list(self.G.out_edges(self.current_node))[action]
        next_node = self.neighbors_sorted[action]

        # TODO: handle dead-end nodes!!
        # if len(edges) < 1:
        #     edges = list(self.G.in_edges(self.current_node))

        cost = self._find_edge_cost((self.current_node, next_node))
        self._set_curr_node_state_info(next_node)

        return self.current_state(), cost, self.current_node == self.dest_node
    
    
    def _set_angle_distance_neighbors_sorted(self):
        distance_fn = lambda x: get_distance(self.G, x, self.dest_node)
        angle_fn = lambda x: get_angle(self.G, x, self.dest_node)

        nbrs_angle_distance_dest_unsorted = list(map(lambda x: (angle_fn(x), distance_fn(x)) ,self.neighbors))

        nbrs_angle_distance_dest = sorted(nbrs_angle_distance_dest_unsorted, key=lambda x:x[0])
        # Padding
        if len(nbrs_angle_distance_dest) < self.MAX_OUT_DEGREE:
            
            pass
        self.neighbors_sorted = nbrs_angle_distance_dest


    def current_state(self):
    
        current_distance_dest = get_distance(self.G, self.current_node, self.dest_node)
        current_angle_dest = get_angle(self.G, self.current_node, self.dest_node)
        
        state = [(self.angle_from_reference, self.distance_from_reference), (current_angle_dest,current_distance_dest)]
        self._set_angle_distance_neighbors_sorted()

        # TODO: use other structre for nbrs to save time
        state+=self.neighbors_sorted
        return state

    def _find_edge_cost(self, edge: tuple):
        # TODO: This may be also a good idea cost = length + haversine distance to target
        # cc = ox.distance.great_circle_vec(*get_lat_lon(self.G, edge[1]), *get_lat_lon(self.G, self.dest_node))
        # return self.G.get_edge_data(edge[0], edge[1])[0]['length'] + cc
        return self.G.get_edge_data(edge[0], edge[1])[0]['length']


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
