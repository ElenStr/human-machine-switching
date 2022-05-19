from collections import defaultdict
from copy import deepcopy
from environments.env import Environment
import networkx
import numpy as np
from math import radians, sin, cos, atan2,degrees
import osmnx as ox
from data.data_utils import get_lat_lon,get_distance
# state: (current node number, destination node number)


# def get_lat_lon(G, node_number):
#     node_dict = G.nodes[node_number]
#     # 'y' is lat, 'x' is lon
#     return [node_dict['y'], node_dict['x']]

# def get_distance(G, node_id_u, node_id_v):
#     """Haversine distance between two nodes"""
    
#     u_coords =  get_lat_lon(G, node_id_u)
#     v_coords =  get_lat_lon(G, node_id_v)
#     distance_km = ox.distance.great_circle_vec(*u_coords,*v_coords)/1000
    
#     return distance_km

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


def trips_list_to_set(graph):
    ret = deepcopy(graph)
    for n in graph.nodes():
        trips_list = graph.nodes[n]['trips']
        ret.nodes[n]['trips'] = set(trips_list)
    return ret





class MapEnv(Environment):
    def __init__(self, graph: networkx.classes.multidigraph.MultiDiGraph, trips):
        self.G = graph
        self.MAX_OUT_DEGREE = max(list(map(lambda x: len(self.G.out_edges(x)), self.G.nodes)))
        
        #  Set reference node to distinuish between nodes
        self.reference_node = 25632226

        self.current_node = None
        self.distance_from_reference = None
        self.angle_from_reference = None
        self.dest_node = None
        self.current_distance_dest = None
        self.current_angle_dest = None
        self.neighbors = []
        self.neighbors_sorted_state = []
        self._define_areas(trips)

        

    def reset(self, id_start, id_end, trip_id):
        """
        Reset the info to retrieve the state to the beginning of new trip 
        """
        self.cur_area = self.trip_areas[trip_id]  
        self.cur_trip = trip_id  
        self.dest_node = id_end
        self._set_curr_node_state_info(id_start)
        self._set_angle_distance_neighbors_sorted()
        
       

    def step(self, action: int):
        """
        returns: state, cost, finished
        """
        # print(f"Action {action}")
        next_node = self.neighbors_sorted[action]

    
        # print(self.current_node)

        prev_curr_node = deepcopy(self.current_node)
        # print("Set current node info")
        self._set_curr_node_state_info(next_node)
        # print("Get next state")

        state = self.current_state()
        # print("Getting cost")
        # print((prev_curr_node, next_node, self.cur_trip))

        cost = self._find_edge_cost((prev_curr_node, next_node), state[1][1])
        
        # TODO check if there are dead-end nodes after cleaning up trips
        dead_end = not len(self.neighbors)
        # print("Set finish")
        finished = (self.current_node == self.dest_node) or dead_end
        
        

        return state, cost, finished
        
    


    def current_state(self):
    
        
        state = [(self.angle_from_reference, self.distance_from_reference), (self.current_angle_dest,self.current_distance_dest)]
        self._set_angle_distance_neighbors_sorted()

        # TODO: use other structre for nbrs to save time
        state+=self.neighbors_sorted_state
        return state

    def next_human_step(self, trip_id):
        """Returns the recorded human driver action and transition"""
        # print("Finding action taken")
        for action, neighbor in enumerate(self.neighbors_sorted):
            
            # print(neighbor, trip_id)
            if trip_id in self.G.nodes[neighbor]['trips']:
                # print("Action found")
                break
        # print("Taking step")
        next_state, cost, finished = self.step(action)
        # print('Step done')
        return action, next_state, cost, finished




    def _set_curr_node_state_info(self, node_id):
        self.current_node = node_id
        # print(node_id)
        self.distance_from_reference = get_distance(self.G, node_id, self.reference_node)
        # print(f"Distance from reference: {self.distance_from_reference}")
        self.angle_from_reference = get_angle(self.G, node_id, self.reference_node)
        # print(f"Angle from reference: {self.angle_from_reference}")

        self.current_distance_dest = get_distance(self.G, self.current_node, self.dest_node)
        self.current_angle_dest = get_angle(self.G, self.current_node, self.dest_node)
        # TODO: check what is the fastest subscriptable structure to use
        self.neighbors = list(self.G.neighbors(node_id))

    def _set_angle_distance_neighbors_sorted(self):
        """Sets self.neighbors sorted"""
         

        distance_fn = lambda x: get_distance(self.G, x, self.dest_node)
        angle_fn = lambda x: get_angle(self.G, x, self.dest_node)

        nbrs_angle_distance_dest_unsorted = list(map(lambda x: [angle_fn(x), distance_fn(x), x] ,self.neighbors))


        # Padding
        padding_size = self.MAX_OUT_DEGREE - len(nbrs_angle_distance_dest_unsorted)
        if  padding_size :
            nbrs_angle_distance_dest_unsorted.extend([(self.current_angle_dest,self.current_distance_dest, self.current_node)]*padding_size)
            
        nbrs_angle_distance_dest = sorted(nbrs_angle_distance_dest_unsorted, key=lambda x:x[0])
        # TODO make more efficient
        self.neighbors_sorted_state = list(map(lambda x:(x[0],x[1]),nbrs_angle_distance_dest))
        self.neighbors_sorted = list(map(lambda x:x[2],nbrs_angle_distance_dest))

    def _find_edge_cost(self, edge: tuple, dist_from_target):
        # if the neighbor is a deadend set an infinite distance to target
        dist_from_target += (self.G.out_degree(edge[1]) == 0)*1000
        dist_to_next = self.G.get_edge_data(edge[0], edge[1])[0]['length']/1000 if edge[0] != edge[1] else 0
        # cost in Km
        return dist_to_next + dist_from_target

# No need to remove dead ends anymore
    def _remove_dead_ends(self, graph):
        done = False
        while not done: 
            dead_ends = [] 
            for id in graph.nodes:
                # find sink nodes 
                if graph.out_degree(id) == 0:
                    dead_ends.append(id)
                    # remove parent of sink that will also become sink
                    for pred in graph.predecessors(id):
                        if graph.out_degree(pred) == 1:
                            dead_ends.append(pred)
            if not len(dead_ends):
                done = True
            for id in dead_ends:
                graph.remove_node(id)
                

    def _define_areas(self, trips_dict):
        distance_fn = lambda x: get_distance(self.G, x, self.reference_node)
        angle_fn = lambda x: get_angle(self.G, x, self.reference_node)

        trips_ids_angle_from_ref_unsorted = list(map(lambda k_v: [angle_fn(k_v[1][1]), distance_fn(k_v[1][1]), k_v[0]] , trips_dict.items()))
        trips_ids_angle_from_ref_sorted = sorted(trips_ids_angle_from_ref_unsorted, key=lambda x:x[0])

        # angles = list(map(lambda x: x[0], trips_ids_angle_from_ref_sorted))
        # trips_ids_dist_ref_sorted = sorted(trips_ids_angle_from_ref_unsorted, key=lambda x:x[1])
        
        # distances = list(map(lambda x: x[1], trips_ids_dist_ref_sorted))
    
        areas = defaultdict(list)
        areas_counts = defaultdict(lambda:0)
        trip_areas = {}

        sectors = 60
        total_trips = len(trips_dict)
        step = total_trips // sectors
        area_idx = -1
        radious_segment = 1000
        area_idys = list(range(radious_segment,8000,radious_segment))
        for angle_idx in range(0,total_trips,  step):
            area_idx+=1
            trips_ids_dist_from_ref_sorted = sorted(trips_ids_angle_from_ref_sorted[angle_idx:(angle_idx+step)], key=lambda x:x[1])
            for _,dist,trip_id in trips_ids_dist_from_ref_sorted:
                for area_idy in area_idys:
                    if dist < area_idy:
                        areas[(area_idx,area_idy)].append(trip_id)
                        areas_counts[(area_idx,area_idy)]+=1
                        trip_areas[trip_id] = (area_idx,area_idy)
                        break
          
        self.areas = areas
        self.areas_counts = areas_counts
        self.trip_areas = trip_areas
       

    @staticmethod
    def state2features(state,n_features, obstacle_to_ignore=''):
        feat =  np.array(state).flatten()
        return feat

# class FeatureHandler:
#     def __init__(self, graph: networkx.classes.multidigraph.MultiDiGraph):
#         self.G = graph
#         self.max_actions = max([self.G.out_degree(n) for n in self.G.nodes])
#         self.feature_size = (self.max_actions + 2) * 2

#     def state2feature(self, state: tuple):
#         """
#         state: (current node number, destination node number)
#         """
#         # current node, destination node, output nodes
#         all_nodes = [state[0]] + [state[1]] + [e[1] for e in self.G.out_edges(state[0])]
#         features = np.array([get_lat_lon(self.G, n) for n in all_nodes]).flatten()
            
#         return np.pad(features, (0, self.feature_size - len(features)), mode='constant')

#     def action_numbers(self, state: tuple):
#         """
#         state: (current node number, destination node number)
#         """
#         return self.G.out_degree(state[0])
