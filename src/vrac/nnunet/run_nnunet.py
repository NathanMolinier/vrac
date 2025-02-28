import os
import argparse
import glob
import subprocess
import json

from vrac.data_management.image import Image

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run nnUNetV2 on folder or config json file.')
    parser.add_argument('--in-files', '-i', required=True, help='Path to the input directory or config json. ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--out-folder', '-o', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet (Required)')
    parser.add_argument('--dataset-number', '-dnum', required=True, type=int, help='Specify the task number.')
    parser.add_argument('--nnunet-plans', '-p', default='', type=str, help='nnUNet plans used for training, if not specified will try to extract automatically')
    parser.add_argument('--nnunet-trainer', '-tr', default='', type=str, help='nnUNet trainer used for training, if not specified will try to extract automatically')
    parser.add_argument('--nnunet-config', '-c', default='', type=str, help='nnUNet configuration used for training, if not specified will try to extract automatically')
    parser.add_argument('--checkpoint', default='checkpoint_best.pth', type=str, help='nnUNet checkpoint, default is checkpoint_best.pth')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load variables
    input_data = os.path.abspath(args.in_files)
    output_folder = os.path.abspath(args.out_folder)
    data_num = args.dataset_number

    # Get list of input files
    if input_data.endswith('.json'):
        with open(input_data, "r") as file:
            config = json.load(file)
        img_list = [os.path.join(config['DATASETS_PATH'], file['IMAGE']) for file in config['TESTING']]
    else:
        img_list = [os.path.join(input_data, file) for file in os.listdir(input_data)]

    # Fetch nnunet paths
    results_path = os.environ["nnUNet_results"]

    # Fetch results folder path
    result_folder = glob.glob(os.path.join(results_path, f'Dataset{data_num}*/*'))

    # Fetch plan and trainer if not specified
    if len(result_folder) > 1 and (not args.nnunet_plans or not args.nnunet_trainer or not args.nnunet_config):
        raise ValueError(f'Multiple results folders detected please specify -p, -c and -tr')
    elif len(result_folder) == 1:
        folder_name = os.path.basename(result_folder[0])
    else:
        for folder in result_folder:
            if args.nnunet_plans in folder and args.nnunet_trainer in folder and args.nnunet_config in folder:
                folder_name = os.path.basename(folder)
    
    plans = args.nnunet_plans if args.nnunet_plans else folder_name.split('__')[1]
    trainer = args.nnunet_trainer if args.nnunet_trainer else folder_name.split('__')[0]
    configuration = args.nnunet_config if args.nnunet_config else folder_name.split('__')[2]

    # Create out directory if does not exist
    in_folder = os.path.join(output_folder, 'input')
    if not os.path.exists(in_folder):
        os.makedirs(in_folder)
    
    pred_folder = os.path.join(output_folder, 'pred')
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    # Reorient and save input files
    for file in img_list:
        Image(file).change_orientation('LPI').save(os.path.join(in_folder, os.path.basename(file).replace('.nii.gz', '_0000.nii.gz')))
    
    # Call nnUNetV2 for inference
    subprocess.check_call([
        "nnUNetv2_predict",
        "-d", str(data_num),
        "-i", in_folder,
        "-o", pred_folder,
        "-f", "0",
        "-c", configuration,
        "-p", plans,
        "-tr", trainer,
        "-chk", args.checkpoint
    ])

if __name__=='__main__':
    main()
