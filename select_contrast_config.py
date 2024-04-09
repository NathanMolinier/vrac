import json
import copy

input_json_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/config_data/sc-seg/can-spine-PSIR-STIR-T1w-SCseg_exclude_balanced.json'
out_json_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/config_data/sc-seg/spinegeneric-T1w-SCseg_exclude_training1.json'

# Read json file and create a dictionary
with open(input_json_path, "r") as file:
    config_data = json.load(file)

new_config_data = copy.copy(config_data)
new_config_data['CONTRASTS'] = 'T1w_T2w'

c = {}
for split in ['TRAINING', 'VALIDATION', 'TESTING']:
    new_data_list = []
    data_paths = config_data[split]
    c[split] = 0
    for d in data_paths:
        if 'T1w' in d['INPUT_IMAGE']:
            new_data_list.append(d)
            c[split] += 1
    new_config_data[split] = new_data_list

print(c)

# Save output json
with open(out_json_path, "w") as file:
    json_object = json.dumps(new_config_data, indent=4)
    file.write(json_object)
