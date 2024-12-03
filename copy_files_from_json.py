import json
import os
import shutil

def main():
    keys_list = ['IMAGE', 'LABEL']
    paths_out_list = ['/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/canal-eval/raw', '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/canal-eval/gt']
    path_json = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/canal-seg/dcm-oklahoma-brno_sci-paris.json'

    with open(path_json, "r") as file:
        config_data = json.load(file)
    
    for dic in config_data['TESTING']:
        for i, key in enumerate(keys_list):
            img_path = os.path.join(config_data['DATASETS_PATH'], dic[key])
            shutil.copy2(img_path, paths_out_list[i])

if __name__=='__main__':
    main()