import json
import os

def main():
    path_json = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/config_data/sc-seg/spinegeneric-T2w-SCseg_exclude.json'
    path_out_txt = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/config_data/sc-seg/spinegeneric-T1w-SCseg_exclude.txt'

    with open(path_json, "r") as file:
        config_data = json.load(file)
    
    dict_list = config_data['TRAINING'] + config_data['VALIDATION'] + config_data['TESTING']

    input_path_list = [os.path.join(config_data['DATASETS_PATH'] ,d['INPUT_LABEL'])+'\n' for d in dict_list]

    with open(path_out_txt, 'w') as file:
        file.writelines(input_path_list)

if __name__=='__main__':
    main()