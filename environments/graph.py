import csv
from collections import defaultdict
import pickle
# from ..environments.env import Environment
import random 
from OSMPythonTools.nominatim import Nominatim

def id_from_coords(coords):
    nominatim = Nominatim()
    node_id = nominatim.query(float(coords[1]), float(coords[0]), reverse=True, zoom=16).id()
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


    def step(self, action):
        pass

    
    def parse(self, file):
        with open(file,'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for i,row in enumerate(csv_reader):
                if i > 0:
                    trip = row[-1][1:-1].split('],[')
                   
                    key_id = None
                    for node_coords in trip:
                        node_coords = node_coords.strip('[').strip(']').split(',')

                        node_id = id_from_coords(node_coords)
                        self.traffic_weights[node_id]+=1
                        if key_id is not None :
                            self.neighbours_list[key_id].add(node_id)
                        key_id = node_id
        
        for k,v in self.neighbours_list.items():
            if len(v) == 1:
                del self.neighbours_list[k]
                del self.traffic_weights[k]

                        
if __name__ == '__main__':
    file = '../train.csv'
    g = Graph()
    # print(g.traffic_weights)
    g.parse(file)
    # print(g.neighbours_list)
    with open('graph.pkl', 'wb') as f:
        pickle.dump(g, f)




                

