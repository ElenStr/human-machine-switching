from csv import writer as csv_writer
from data.data_utils import *
from definitions import ROOT_DIR



def corrupted_trips(trips_path, G, ref_G):
    trips_m_finish = trips_mutli_finish(trips_path)
    

    trips_missing_nodes, missing_nodes = trips_with_missing_nodes(trips_path, G, ref_G)

    trips_to_remove = sorted(trips_m_finish + trips_missing_nodes)
 

    return set(trips_to_remove)


def write_valid_trips(new_trips_path, old_trips_path, graph, ref_graph, valid_fn):
    trips_to_rm = valid_fn(old_trips_path, graph, ref_graph)
    j = 0
    id_rm_start = False
    # id_rm_finish = False
    with open(new_trips_path, 'w') as fw:
        writer = csv_writer(fw)
        with open(old_trips_path, 'r') as fr:
            reader = csv_reader(fr)

            for i,row in enumerate(reader):
                if i > 0:
                    trip_id = float(row[4])
                    if trip_id != trips_to_rm[j]:
                        if not id_rm_start:
                            writer.writerow(row)
                        else:
                            if j < len(trips_to_rm)-1:
                                if trip_id != trips_to_rm[j+1]:
                                    id_rm_start = False 
                                    writer.writerow(row)
                                else:
                                    pass
                                j+=1

                            else:
                                id_rm_start = False 
                                writer.writerow(row)

                    else:
                        if not id_rm_start:
                            id_rm_start = True
                        else:
                            pass
                else:
                    # write header
                    writer.writerow(row)



if '__name__'=='__main__':
    trips_no_mfinish_no_missing_nodes = './data/no_mfin_no_mis_nodes.csv'
    old_trips = './processed_trips.csv'
    final_trips_path='data/cleaned_up_trips.csv'
    graph_path = 'data/Porto_driving.osm'
    G = ox.graph_from_xml(graph_path)
    ref_G = ox.graph_from_place('Distrito do Porto, PT', clean_periphery=False,network_type='drive')


    write_valid_trips(new_trips_path=trips_no_mfinish_no_missing_nodes, old_trips_path=old_trips, graph=G, ref_graph=ref_G, valid_fn=corrupted_trips)

    graph_with_no_missing_nodes, has_dead_ends = graph_from_trips(trips_no_mfinish_no_missing_nodes,G, ref_G)
    tmp_graph = graph_with_no_missing_nodes
    tmp_path = trips_no_mfinish_no_missing_nodes 
    while not has_dead_ends:

        write_valid_trips(final_trips_path,tmp_path,graph=tmp_graph, ref_graph=ref_G, valid_fn=trips_with_dead_ends)

        final_graph, has_dead_ends = graph_from_trips(final_trips_path, tmp_graph, ref_G)
        tmp_graph = final_graph
        tmp_path = final_trips_path



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
                         








                        




