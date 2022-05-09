from definitions import ROOT_DIR
from csv import reader as csv_reader
import osmnx as ox
from copy import deepcopy

def get_lat_lon(graph, node_number):
    node_dict = graph.nodes[node_number]
    # 'y' is lat, 'x' is lon
    return [node_dict['y'], node_dict['x']]

def get_distance(graph, node_id_u, node_id_v):
    """Haversine distance between two nodes"""
        
    u_coords =  get_lat_lon(graph, node_id_u)
    v_coords =  get_lat_lon(graph, node_id_v)
    distance_km = ox.distance.great_circle_vec(*u_coords,*v_coords)
    
    return distance_km



def trips_mutli_finish(processed_trip_path):
    with open(processed_trip_path, 'r') as f:
        reader = csv_reader(f)
        trips_mfinish = []
        all_trips = []
        check = False
        for i,row in enumerate(reader):
            # if finished
            if row[2] == 'True':
                trip_id = float(row[4])
                # First trip that finishes
                if not len(all_trips):
                    # Check if it finishes again
                    check = True
                    all_trips.append(trip_id)
                else:
                    if all_trips[-1] != trip_id:
                        check = True
                        all_trips.append(trip_id)
                    else:
                        if check:
                            trips_mfinish.append(trip_id)
                            # Prevent from double counting if it fnishes again
                            check = False
    return trips_mfinish




def trips_with_missing_nodes(processed_trip_path, proc_graph, ref_graph):
    graph = deepcopy(proc_graph)
    with open(processed_trip_path, 'r') as f:
        reader = csv_reader(f)
        trips_to_exclude = []
        nodes_to_exclude = []
        for i,row in enumerate(reader):
            if i > 0:
                trip_id = float(row[4])
                u = int(row[0])
                v = int(row[1])
                if (not len(trips_to_exclude)) or trip_id!=trips_to_exclude[-1]:      
                    for node in [u,v]:
                        try:                
                            graph.nodes[node]
                        except KeyError:
                            try:
                                ref_graph.nodes[node]
                                graph.add_node(node)
                                for key,val in ref_graph.nodes[node].items():
                                    graph.nodes[node][key] = val
                                
                                if not graph.has_edge(u,v):
                                    length = row[3]
                                    if length == '':
                                        length = get_distance(ref_graph,u,v)
                                    else:
                                        length = float(length)
                                        
                                    graph.add_edge(u,v, length=length)
                            except KeyError:
                                if (not len(trips_to_exclude)) or trip_id!=trips_to_exclude[-1]:
                                    trips_to_exclude.append(trip_id)
                                    nodes_to_exclude.append(node)
                                    
                                   

                        
        return trips_to_exclude,list(set(nodes_to_exclude))
                    
def graph_from_trips(trips_path, graph, ref_graph):
    """Return a graph that corresponds to the trips_path"""
    with open(trips_path, 'r') as f:
        reader = csv_reader(f)
        for i,row in enumerate(reader):
            if i > 0:
                trip_id = float(row[4])
                u = int(row[0])
                v = int(row[1])
                
                for node in [u,v]:
                    try:                
                        graph.nodes[node]
                    except KeyError:
                        try:
                            ref_graph.nodes[node]
                            graph.add_node(node)
                            for key,val in ref_graph.nodes[node].items():
                                graph.nodes[node][key] = val
                            
                        except KeyError:
                            graph.remove_node(node)
                    try:
                        if not graph.has_edge(u,v):
                            length = row[3]
                            if length == '':
                                length = get_distance(ref_graph,u,v)
                            else:
                                length = float(length)
                                
                            graph.add_edge(u,v, length=length)
                    except KeyError:
                        graph.remove_node(u)
                        graph.remove_node(v)
            
    return graph


                        

                
def add_trip_ids_to_nodes(processed_trip_path,graph):
    # initialize trip id lists
    for id in graph.nodes:
        graph.nodes[id]['trips'] = []

    with open(processed_trip_path, 'r') as f:
        reader = csv_reader(f)
        k = 0
        for i,row in enumerate(reader):
            if i > 0:
                trip_id = float(row[4])
                u = int(row[0])
                v = int(row[1])
                
                try:                
                    graph.nodes[u]['trips'].append(trip_id)
                    graph.nodes[v]['trips'].append(trip_id)
                except KeyError:
                    k+=1
        print(k,i)
    return graph


def trips_with_dead_ends(valid_graph):
    graph = deepcopy(valid_graph)
    done = False
    trips_with_dead_ends = []
    while not done: 
        dead_ends = [] 
        for id in graph.nodes:
            # find sink nodes 
            if graph.out_degree(id) == 0:
                dead_ends.append(id)
                trips_with_dead_ends.extend(graph.nodes[id]['trips'])
                # remove parent of sink that will also become sink
                for pred in graph.predecessors(id):
                    if graph.out_degree(pred) == 1:
                        dead_ends.append(pred)
                        trips_with_dead_ends.extend(graph.nodes[pred]['trips'])

        if not len(dead_ends):
            done = True
        for id in dead_ends:
            graph.remove_node(id)
    return trips_with_dead_ends
