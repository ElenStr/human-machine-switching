import csv
from collections import defaultdict
import os
import pickle
import osmnx as ox
# from ..environments.env import Environment
import random 
from OSMPythonTools.nominatim import Nominatim
import concurrent.futures
import threading
queries_lock = threading.Lock()

def id_from_coords(coords):
    nominatim = Nominatim()
    with queries_lock:
        node_id = nominatim.query(float(coords[1]), float(coords[0]), reverse=True, zoom=16, waitBetweenQueries=0.5).id()
    os.rmdir('../cache')
    return node_id

def crossroad(node_id):
    return 1
def list_init():
    return set()
def zero_init():
    return 0


class Graph():
    """ The environment object"""
    def __init__(self) -> None:
        self.neighbours_list = defaultdict(list_init)
        self.traffic_weights = defaultdict(zero_init)
        self.traffic_lock = threading.Lock()
        self.neighbour_lock = threading.Lock()

    def step(self, action):
        pass

    
    def parse(self, file):
        def parse_trip(trip):
            key_id = None
            for node_coords in trip:
                node_coords = node_coords.strip('[').strip(']').split(',')

                node_id = id_from_coords(node_coords)
                with self.traffic_lock:
                    self.traffic_weights[node_id]+=1

                if key_id is not None :
                    with self.neighbour_lock:
                        self.neighbours_list[key_id].add(node_id)
                key_id = node_id

        with open(file,'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            trips = []
            for i,row in enumerate(csv_reader):
                if i > 0 and i < 2:
                    trip = row[-1][1:-1].split('],[')
                    trips.append(trip)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(trips)) as executor:
            executor.map(parse_trip, trips)
        
        for k,v in self.neighbours_list.items():
            if len(v) == 1:
                del self.neighbours_list[k]
                del self.traffic_weights[k]

                        
if __name__ == '__main__':
    file = '../train.csv'
    place = "Porto, Portugal"
    graph = ox.graph_from_place(place, network_type='drive')
    print(graph.order())
    # print(g.traffic_weights)
    # g.parse(file)
    # print(g.neighbours_list)
    # with open('graph.pkl', 'wb') as f:
    #     pickle.dump(g.neighbours_list, f)
    # with open('traffics.pkl', 'wb') as f:
    #     pickle.dump(g.traffic_weights, f)




                

