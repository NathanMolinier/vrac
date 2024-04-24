"""
This script adds spine and spinal cord (SC) segmentations to our private whole-spine dataset

https://github.com/neuropoly/data-management/issues/308
"""

import json
import os
import numpy as np
import subprocess
import time

from vrac.data_management.image import Image, zeros_like
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    input_folder = '/Users/nathan/data/whole-spine/derivatives/whole_totalsegmri'
    bids_folder = '/Users/nathan/data/whole-spine'
    json_mapping = '/Users/nathan/data/whole-spine/derivatives/whole_totalsegmri/whole-spine_mapping_old_new_names.json'
    derivatives_path = os.path.join(bids_folder, 'derivatives/labels')
    qc_path = os.path.join(bids_folder, 'qc')
    sc_value = 200

    # Open json
    with open(json_mapping, "r") as file:
        map_names = json.load(file)
    
    err = []
    missing_files = []
    for old_name, new_name in map_names.items():
        for cont in ['T1w', 'T2w']:
            fname = f'whole_{old_name.replace('sub-','')}_{cont}.nii.gz'
            file_path = os.path.join(input_folder, fname)
            if not os.path.exists(file_path):
                print(f'File {fname} is missing !')
                missing_files.append(fname)
            else:
                # Init new file paths
                sc_mask_name = f'{new_name}_{cont}_label-SC_seg.nii.gz'
                sc_mask_path = os.path.join(derivatives_path, new_name, 'anat', sc_mask_name)
                
                spine_seg_name = f'{new_name}_{cont}_label-spine_dseg.nii.gz'
                spine_seg_path = os.path.join(derivatives_path, new_name, 'anat', spine_seg_name)
                
                # Get image path for QC
                img_path = get_img_path_from_label_path(sc_mask_path)
                img = Image(img_path).change_orientation('RPI')

                # Load file
                spine_label = Image(file_path).change_orientation('RPI')
                sc_label = zeros_like(spine_label)
                sc_label.data[np.where(spine_label.data == sc_value)] = 1

                # Check if equal size
                if img.dim != sc_label.dim or img.dim != spine_label.dim:
                    print(f'Size mismatch between {img_path} and {sc_mask_path}')
                    err.append(img_path)

                # Create or edit json sidecars
                create_json_file(spine_seg_path.replace('.nii.gz', '.json'))
                create_json_file(sc_mask_path.replace('.nii.gz', '.json'))

                # Save files using RPI orientation
                spine_label.change_orientation('RPI').save(spine_seg_path)
                sc_label.change_orientation('RPI').save(sc_mask_path)
                img.change_orientation('RPI').save(img_path)

                # QC images
                subprocess.check_call([
                    "sct_qc", 
                    "-i", img_path,
                    "-s", sc_mask_path,
                    "-p", "sct_deepseg_sc",
                    "-qc", qc_path

                ])
    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("err files:\n" + '\n'.join(sorted(err)))


def create_json_file(path_json_out):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    if path_json_out.endswith('.nii.gz'):
        raise ValueError('Path to image specified ! Use .json extension !')
    
    if os.path.exists(path_json_out):
        with open(path_json_out, 'r') as file:
            config_dict = json.load(file)
            print(f'JSON {path_json_out} was loaded')
    
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "Manual",
                "Author": "Yehuda Warszawer",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                "Name": "Manual",
                "Author": "Nathan Molinier",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }
    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Created: {path_json_out}')

if __name__=='__main__':
    main()
            
            

            