from vrac.data_management.image import Image
import os, json, argparse
from progress.bar import Bar
import numpy as np

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Copy data config to folder with resampling and reorientation.')
    parser.add_argument('--config', '-c', required=True, help='Path to the config json file. ~/<your_path>/config_data.json (Required)')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load variables
    config_raw_path = args.config

    # Load json config
    with open(config_raw_path, "r") as file:
        config = json.load(file)
    
    sizes = []
    res = []
    
    for split in ['TRAINING', 'VALIDATION', 'TESTING']:
        dict_list = config[split]
        bar = Bar(f'Deal with {split} data', max=len(dict_list))
        for dic in dict_list:
            # Init paths
            label_path = os.path.join(config['DATASETS_PATH'], dic['LABEL'])

            # Load data
            label = Image(label_path)

            # Reorient data to RSP
            label.change_orientation('RSP')
            
            sizes.append(list(label.dim[:3]))
            res.append(list(label.dim[4:7]))

            # Plot progress
            bar.suffix  = f'{dict_list.index(dic)+1}/{len(dict_list)}'
            bar.next()
        bar.finish()
    
    sizes = np.array(sizes)

    print(f'Right-Left:\n min = {np.min(sizes[:,0])}\n max = {np.max(sizes[:,0])}\n median = {np.median(sizes[:,0])}\n\n')
    print(f'Superior-Inferior:\n min = {np.min(sizes[:,1])}\n max = {np.max(sizes[:,1])}\n median = {np.median(sizes[:,1])}\n\n')
    print(f'Anterior-Posterior:\n min = {np.min(sizes[:,2])}\n max = {np.max(sizes[:,2])}\n median = {np.median(sizes[:,2])}\n\n')


if __name__=='__main__':
    main()