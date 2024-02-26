import json
import pandas as pd

def main():
    path_old_tsv="/Users/nathan/Desktop/totalsegmri_whole-spine_participants.tsv"
    path_new_tsv="/Users/nathan/Desktop/participants.tsv"
    output_json_path = "/Users/nathan/Desktop/mapping_old_new.json"

    old_dict = create_name_maps_participant_tsv(path_old_tsv)
    new_dict = create_name_maps_participant_tsv(path_new_tsv)
    
    # Sort by values see https://realpython.com/sort-python-dictionary/
    old_dict = dict(sorted(old_dict.items(), key=lambda item: item[1]))

    map_old_new = {}
    for k, v in old_dict.items():
        map_old_new[v] = new_dict[k]

    with open(output_json_path, 'w') as out:
        json_object = json.dumps(map_old_new, indent=4)
        out.write(json_object)



def create_name_maps_participant_tsv(tsv_path):
    name_maps = {}
    tsv_dict = pd.read_csv(tsv_path, sep='\t').to_dict()
    for num, data_id in tsv_dict['data_id'].items():
        name_maps[data_id] = tsv_dict['participant_id'][num]
    return name_maps

if __name__=='__main__':
    main()