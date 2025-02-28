from vrac.data_management.image import Image
import argparse
import os, json

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Copy config split to folder.')
    parser.add_argument('--config', '-c', required=True, help='Path to the config json file. ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--split', '-sp', required=True, type=str, help='Config split. Example: TRAINING (Required)')
    parser.add_argument('--type', '-t', required=True, type=str, help='Which image to copy. Example: IMAGE, LABEL... (Required)')
    parser.add_argument('--out-folder', '-o', required=True, type=str, help='Path to the output folder.')
    parser.add_argument('--reorient', default='LPI', type=str, help='Orientation of the final copied images. Default is LPI.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load json config
    with open(args.config, "r") as file:
        config = json.load(file)

    # Load variables
    output_folder = os.path.abspath(args.out_folder)
    split = args.split
    reorient = args.reorient
    img_type = args.type

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for dic in config[split]:
        img = Image(os.path.join(config['DATASETS_PATH'], dic[img_type]))
        img.change_orientation(reorient)
        img.save(os.path.join(output_folder, os.path.basename(dic[img_type])))

if __name__=='__main__':
    main()