import os
import subprocess
import glob

def main():
    data_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/sexy-images/raw'
    dataset_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets' # Folder where datasets will be downloaded
    map_key_dataset = {
        "basel":"basel-mp2rage",
        "brno":"dcm-brno",
        "canproco":"canproco",
        "large":"sct-testing-large",
        "marseille":"marseille-3t-mp2rage",
        "multi":"data-multi-subject",
        "nusantara":"lumbar-nusantara",
        "oklahoma":"dcm-oklahoma",
        "pediatric":"philadelphia-pediatric",
        "ulrike":"hc-leipzig-7t-mp2rage",
        "vanderbilt":"lumbar-vanderbilt",
        "whole":"whole-spine"
    }
    out_text = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/vert-labeling/sexy_data.txt'
    txt_list = []
    for fname in os.listdir(data_folder):
        split_name = fname.split('_')
        keyword = split_name[0]
        dataset = map_key_dataset[keyword]
        img_name = "_".join(split_name[1:])
        rep_path = os.path.join(dataset_folder, dataset)

        if not os.path.exists(rep_path):
            subprocess.check_call([
                    'git', 'clone', f'git@data.neuro.polymtl.ca:datasets/{dataset}', rep_path
                ])
        
        glob_path = glob.glob(f'{rep_path}/**/{img_name}', recursive=True)
        if len(glob_path) != 1:
            raise ValueError('Abnormal number of paths detected')
        else:
            txt_list.append(glob_path[0]+'\n')
    
    with open(out_text, 'w') as f:
        f.writelines(txt_list)
        
    print()


if __name__=='__main__':
    main()