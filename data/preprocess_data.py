from csv import writer as csv_writer
from data.data_utils import *
from definitions import ROOT_DIR


def trips_to_remove_save_filled_graph(trips_path, graph_path):
    trips_m_finish = trips_mutli_finish(trips_path)
    G = ox.graph_from_xml(graph_path)
    ref_G = ox.graph_from_place('Distrito do Porto, PT', clean_periphery=False)

    trips_with_unknown_nodes,G = fill_graph(trips_path, G, ref_G)

    ox.io.save_graphml(G, filepath=f"{ROOT_DIR}/data/filled_graph_porto.osm")
    G = add_trip_ids_to_nodes(trips_path,G)


    trips_with_dead_ends,G = list(set(trips_dead_ends(G)))

    return trips_m_finish+trips_with_unknown_nodes+trips_with_dead_ends


def write_valid_trips(new_trips_path, old_trips_path, graph_path):
    trips_to_rm = sorted(trips_to_remove_save_filled_graph(old_trips_path, graph_path))
    j = 0
    id_rm_start = False
    # id_rm_finish = False
    with open(new_trips_path, 'w') as fw:
        writer = csv_writer(fw)
        with open(old_trips_path, 'r') as fr:
            reader = csv_reader(fr)

            for i,row in enumerate(reader):
                if i > 0:
                    trip_id = row[4]
                    if trip_id != trips_to_rm[j]:
                        if not id_rm_start:
                            writer.writerow(row)
                        else:
                            if j < len(trips_to_rm):
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
                    pass
                         








                        




