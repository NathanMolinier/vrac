"""
The aim of this script is to reorient and save segmentations generated using active learning inside the derivatives/labels folder. The JSON files have to be updated accordingly too.
"""

import os, argparse, json
from vrac.data_management.image import Image, zeros_like
import time 
import numpy as np
import copy
import subprocess

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Bidsify spine and canal segmentations.')
    parser.add_argument('-img', required=True, help='Path to the image folder containing images of subjects that will be saved (Required)')
    parser.add_argument('-pred', required=True, help='Path to the prediction folder containing segmentations of TotalSpineSeg (Required)')
    parser.add_argument('-ofolder', required=True, help='Path to the output BIDS folder (Required)')
    return parser

def create_json_file(path_json_out):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """

    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "TotalSpineSeg",
                "Description": "Segmentation produced using an active learning strategy.",
                "Link": "https://github.com/neuropoly/totalspineseg/blob/bb90ac3dca5f856f4cd71c65467efa023be444cf/scripts/active_learning/README.md",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }
    
    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'JSON {path_json_out} was created.')

def main(img_folder, pred_folder, output_folder):
    # List paths
    img_list = os.listdir(img_folder)
    err = []
    spine_list = []
    # Deal with spine segmentations
    for img_file in img_list:
        if img_file.startswith('sub-') and img_file.endswith('.nii.gz'):
            subj = img_file.split('_')[0]
            contrast = img_file.split('_')[1].replace('.nii.gz', '')

            img_path = os.path.join(output_folder, subj, 'anat', img_file)
            seg_path = os.path.join(pred_folder, img_file)

            # Load image
            seg = Image(seg_path).change_orientation('RPI')
            img = Image(img_path).change_orientation('RPI')

            # Check if equal size
            if img.dim[:3] != seg.dim[:3]:
                err.append(seg_path)
            else:
                # Create spine and canal segmentations
                spine_seg = copy.deepcopy(seg)
                canal_seg = zeros_like(seg)

                canal_seg.data[np.isin(seg.data, [1,2])] = 1
                spine_seg.data[np.isin(seg.data, [1,2])] = 0

                # Init path out
                output_path_canal = os.path.join(output_folder, 'derivatives/labels', subj, "anat", f'{subj}_{contrast}_label-canal_seg.nii.gz')
                output_path_spine = os.path.join(output_folder, 'derivatives/labels', subj, "anat", f'{subj}_{contrast}_label-spine_dseg.nii.gz')
                output_path_canal_json = output_path_canal.replace('.nii.gz', '.json')
                output_path_spine_json = output_path_spine.replace('.nii.gz', '.json')

                # Check existing json sidecars
                if os.path.exists(output_path_canal_json):
                    with open(output_path_canal_json, 'r') as f:
                        data_json = json.load(f)
                    
                    skip = False
                    for dic in data_json['GeneratedBy']:
                        if 'Author' in dic.keys():
                            if dic['Author'] == 'Abel Salmona':
                                skip = True
                        if 'Author' in dic.keys():
                            if dic['Author'] == 'Sandrine Bedard':
                                skip = True
                        if 'Name' in dic.keys():
                            if 'spinalcordtoolbox: sct_deepseg sc_canal_t2' in dic['Name']:
                                skip = True
                else:
                    skip = False
                
                if not skip:
                    # Save segmentations
                    canal_seg.save(output_path_canal)

                    # Create JSON files
                    create_json_file(output_path_canal_json)
                
                # Save segmentations
                spine_seg.save(output_path_spine)

                # Create JSON files
                create_json_file(output_path_spine_json)

                # QC canal and spine segmentations
                subprocess.run(['sct_qc', '-i', img_path, '-s', output_path_canal, '-p', 'sct_deepseg_sc', '-qc', os.path.join(output_folder, 'derivatives/labels', 'qc-canal')])
                subprocess.run(['sct_qc', '-i', img_path, '-s', output_path_spine, '-p', 'sct_label_vertebrae', '-qc', os.path.join(output_folder, 'derivatives/labels', 'qc-spine')])

                # Save spine file in txt file
                spine_list.append(output_path_spine)

    # Print all error files
    print(f'Errors (size mismatch) for {len(err)} files:')
    for e in err:
        print(f' - {e}')
    
    # Save all spine files in a txt file
    with open(os.path.join(output_folder, 'derivatives/labels', 'spine_list.txt'), 'w') as f:
        for item in spine_list:
            f.write("%s\n" % item)

if __name__=='__main__':
    # Load parser
    parser = get_parser()
    args = parser.parse_args()

    # Define paths
    img_folder = args.img
    pred_folder = args.pred
    output_folder = args.ofolder
    # img_folder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/tss_active/wholespine/t1w'
    # pred_folder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/tss_active/wholespine/out-t1w/step2_output'
    # output_folder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/whole-spine'
    main(img_folder, pred_folder, output_folder)