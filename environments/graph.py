import csv
from collections import defaultdict
from environments.env import Environment


class Graph(Environment):
    """ The environment object"""
    def __init__(self) -> None:
        self.neighbours_list = defaultdict(lambda: [])
        self.traffic_weights = defaultdict(lambda: 0)


    def step(self, action):
        pass

    @staticmethod
    def parse(file):
        with open(file,'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for i,row in enumerate(csv_reader):
                if i > 0:
                    trip = row[-1]
                    key_id = None
                    for node_coords in trip:
                        node_id = id_from_coords(node_coords)
                        if crossroad(node_id):
                            Graph.traffic_weights[node_id]+=1
                            if key_id is not None :
                                Graph.neighbours_list[key_id].append(node_id)
                            key_id = node_id
                        
                        



                

