from asyncore import ExitNow
from csv import writer as csv_writer
from data.data_utils import *
from definitions import ROOT_DIR
from datetime import datetime


def get_cur_time_str():
    now = datetime.now()

    return f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"


def corrupted_trips(trips_path, G, ref_G):
    trips_m_finish = trips_mutli_finish(trips_path)
    

    trips_missing_nodes, missing_nodes = trips_with_missing_nodes(trips_path, G, ref_G)

    trips_to_remove = trips_m_finish + trips_missing_nodes
 

    return sorted(list(set(trips_to_remove)))


def write_valid_trips(new_trips_path, old_trips_path, graph, ref_graph, valid_fn):
    print(f"{get_cur_time_str()} Gathering trips to remove...")
    trips_to_rm = valid_fn(old_trips_path, graph, ref_graph)
    print(f"{get_cur_time_str()} {len(trips_to_rm)} trips to remove")

    print(f"{get_cur_time_str()} Copy and remove invalid trips...")
    j = 0
    id_rm_start = False
    trips_to_rm_cnt = len(trips_to_rm)
    # id_rm_finish = False
    with open(new_trips_path, 'w') as fw:
        writer = csv_writer(fw)
        with open(old_trips_path, 'r') as fr:
            reader = csv_reader(fr)
            header = next(reader)
            writer.writerow(header)
            try:
                row = next(reader)
                trip_id = int(float(row[4]))
                while j < trips_to_rm_cnt:

                    while trip_id != trips_to_rm[j]:
                        # if trip_id == 5:
                            # print(trips_to_rm[j])
                        writer.writerow(row)
                        row = next(reader)
                        trip_id = int(float(row[4]))
                    # print(f"Removing trip {row}")
                    while trip_id == trips_to_rm[j]:
                        row = next(reader)
                        trip_id = int(float(row[4]))
                        # if trip_id == 7:
                            # print('NOOOOO')
                    # print(f"Next trip to check{row}")
                    j+=1

                while row:
                    writer.writerow(row)
                    row = next(reader)
            
            except StopIteration:
                # print(row, trips_to_rm[j])
                pass


    print(f"{get_cur_time_str()} trips removed {j}")



if __name__=='__main__':
    trips_no_mfinish_no_missing_nodes = './data/no_mfin_no_mis_nodes.csv'
    old_trips = './processed_trips.csv'
    final_trips_path='data/cleaned_up_trips.csv'
    graph_path = 'data/Porto_driving.osm'
    
    print('Reading graph')
    G = ox.graph_from_xml(graph_path)

    print(f"{get_cur_time_str()} Downloading Porto Graph...")
    ref_G = ox.graph_from_place('Distrito do Porto, PT', clean_periphery=False,network_type='drive')

    # print(f"{get_cur_time_str()} Writing single finish trips with known nodes")
    # write_valid_trips(new_trips_path=trips_no_mfinish_no_missing_nodes, old_trips_path=old_trips, graph=G, ref_graph=ref_G, valid_fn=corrupted_trips)
    print(f"{get_cur_time_str()} Syncing graph to trips...")
    
    graph_with_no_missing_nodes, has_dead_ends = graph_from_trips(trips_no_mfinish_no_missing_nodes,G, ref_G)
    print(has_dead_ends)

    tmp_graph = graph_with_no_missing_nodes
    tmp_path = trips_no_mfinish_no_missing_nodes  
    # TODO: works in our case, but make it work in the general case (new dead ends may appear every time)
    if has_dead_ends:

        write_valid_trips(final_trips_path,tmp_path,graph=tmp_graph, ref_graph=ref_G, valid_fn=trips_with_dead_ends)

        final_graph, has_dead_ends = graph_from_trips(final_trips_path, tmp_graph, ref_G)

        
        save_osm_graph(final_graph, "./data/final_graph.som")
        
        print(has_dead_ends, final_graph.number_of_nodes(), final_graph.number_of_edges() )
       



# def trips_to_remove_save_filled_graph(trips_path, graph_path):
#     trips_m_finish = trips_mutli_finish(trips_path)
#     G = ox.graph_from_xml(graph_path)
#     ref_G = ox.graph_from_place('Distrito do Porto, PT', clean_periphery=False,network_type='drive')

#     trips_with_unknown_nodes,G = fill_graph(trips_path, G, ref_G)

#     ox.io.save_graphml(G, filepath=f"{ROOT_DIR}/data/filled_graph_porto.osm")
#     G = add_trip_ids_to_nodes(trips_path,G)


#     trips_with_dead_ends = list(set(trips_dead_ends(G)))

#     return trips_m_finish+trips_with_unknown_nodes+trips_with_dead_ends


# def write_valid_trips(new_trips_path, old_trips_path, graph_path):
#     trips_to_rm = sorted(trips_to_remove_save_filled_graph(old_trips_path, graph_path))
#     j = 0
#     id_rm_start = False
#     # id_rm_finish = False
#     with open(new_trips_path, 'w') as fw:
#         writer = csv_writer(fw)
#         with open(old_trips_path, 'r') as fr:
#             reader = csv_reader(fr)

#             for i,row in enumerate(reader):
#                 if i > 0:
#                     trip_id = row[4]
#                     if trip_id != trips_to_rm[j]:
#                         if not id_rm_start:
#                             writer.writerow(row)
#                         else:
#                             if j < len(trips_to_rm):
#                                 if trip_id != trips_to_rm[j+1]:
#                                     id_rm_start = False 
#                                     writer.writerow(row)
#                                 else:
#                                     pass
#                                 j+=1

#                             else:
#                                 id_rm_start = False 
#                                 writer.writerow(row)

#                     else:
#                         if not id_rm_start:
#                             id_rm_start = True
#                         else:
#                             pass
#                 else:
#                     pass
                         








                        




