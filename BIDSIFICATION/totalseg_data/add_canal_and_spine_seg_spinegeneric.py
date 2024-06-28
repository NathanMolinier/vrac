"""
This script extracts the canal and the spine segmentations from the totalseg multi class segmentations and adds them to spinegeneric datasets.

"""

import json
import os
import numpy as np
import subprocess
import shutil
import time
from copy import copy

from vrac.data_management.image import Image, zeros_like
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    input_folder = '/Users/nathan/Desktop/totalseg_data/multi'
    bids_folder = '/Users/nathan/data/data-multi-subject'
    derivatives_path = os.path.join(bids_folder, 'derivatives/labels')
    qc_path = os.path.join(bids_folder, 'qc')
    sc_value = [200]
    csf_value = [201]
    canal_value = sc_value + csf_value
    
    err = []
    missing_files = []
    for sub in os.listdir(bids_folder):
        if sub.startswith('sub'):
            for cont in ['T1w', 'T2w']:
                raw_name = f'{sub}_{cont}_totalsegmri.nii.gz'
                raw_path = os.path.join(input_folder, 'derivatives/labels', sub, 'anat', raw_name)
                if not os.path.exists(raw_path):
                    print(f'File {raw_path} is missing !')
                    missing_files.append(raw_path)
                else:
                    # Init new file paths
                    canal_seg_name = f'{sub}_{cont}_label-canal_seg.nii.gz'
                    canal_seg_path = os.path.join(derivatives_path, sub, 'anat', canal_seg_name)

                    sc_seg_name = f'{sub}_{cont}_label-SC_seg.nii.gz'
                    sc_seg_path = os.path.join(derivatives_path, sub, 'anat', sc_seg_name)
                    
                    spine_seg_name = f'{sub}_{cont}_label-spine_dseg.nii.gz'
                    spine_seg_path = os.path.join(derivatives_path, sub, 'anat', spine_seg_name)
                    
                    # Get image path for QC
                    img_path = get_img_path_from_label_path(spine_seg_path)
                    img = Image(img_path).change_orientation('RPI')

                    # Load file
                    totalseg_dseg = Image(raw_path).change_orientation('RPI')

                    # Check if equal size
                    if img.dim[:3] != totalseg_dseg.dim[:3]:
                        raise ValueError(f'Size mismatch between {img_path} and {raw_path}')
                        err.append(img_path)
                    
                    # Create spinal cord segmentation if does not exist
                    if not os.path.exists(sc_seg_path):
                        sc_seg = zeros_like(totalseg_dseg)
                        for value in sc_value:
                            sc_seg.data[np.where(totalseg_dseg.data == value)] = 1

                    # Create canal array
                    canal_seg = zeros_like(totalseg_dseg)
                    for value in canal_value:
                        canal_seg.data[np.where(totalseg_dseg.data == value)] = 1

                    # Create spine array
                    spine_dseg = zeros_like(totalseg_dseg)
                    spine_dseg.data = copy(totalseg_dseg.data)
                    for value in canal_value:
                        spine_dseg.data[np.where(totalseg_dseg.data == value)] = 0

                    # Copy and edit json sidecars
                    path_in_json = raw_path.replace('.nii.gz', '.json')
                    path_out_json_spine = spine_seg_path.replace('.nii.gz', '.json')
                    path_out_json_canal = canal_seg_path.replace('.nii.gz', '.json')
                    shutil.copy(path_in_json, path_out_json_spine)
                    shutil.copy(path_in_json, path_out_json_canal)
                    edit_json_file(path_out_json_spine)
                    edit_json_file(path_out_json_canal)

                    # Save files using RPI orientation
                    canal_seg.change_orientation('RPI').save(canal_seg_path)
                    spine_dseg.change_orientation('RPI').save(spine_seg_path)
                    # img.change_orientation('RPI').save(img_path)

                    if not os.path.exists(sc_seg_path):
                        # Add sc only if segmentation does not exist
                        path_out_json_sc = sc_seg_path.replace('.nii.gz', '.json')
                        shutil.copy(path_in_json, path_out_json_sc)
                        edit_json_file(path_out_json_sc)
                        sc_seg.change_orientation('RPI').save(sc_seg_path)
                        subprocess.check_call([
                            "sct_qc", 
                            "-i", img_path,
                            "-s", sc_seg_path,
                            "-p", "sct_deepseg_sc",
                            "-qc", qc_path
                        ])

                    # QC images
                    subprocess.check_call([
                        "sct_qc", 
                        "-i", img_path,
                        "-s", canal_seg_path,
                        "-p", "sct_deepseg_sc",
                        "-qc", qc_path
                    ])

                    subprocess.check_call([
                        "sct_qc", 
                        "-i", img_path,
                        "-s", spine_seg_path,
                        "-d", spine_seg_path,
                        "-p", "sct_deepseg_lesion",
                        "-plane", "sagittal",
                        "-qc", qc_path
                    ])
                    
    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("err files:\n" + '\n'.join(sorted(err)))


def edit_json_file(path_json_out):
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
                "Name": "sct_apply_transfo",
                "Version": "SCT v6~",
                "Date": config_dict["Date"],
                "Description": "Subject registration to the PAM50 template then reverse transformation applied" 
                " to the template to bring the segmentations/atlas to the subject space."
                " see https://github.com/neuropoly/totalspineseg/blob/main/run/reg2pam50.sh"
            },
            {
                "Name": "Manual",
                "Author": "Yehuda Warszawer",
                "Date": config_dict["Date"]
            },
            {
                "Name": "Manual",
                "Author": "Nathan Molinier",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }

    sc_data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "sct_deepseg_sc",
                "Kernel": "2d",
                "Version": "SCT v6~",
                "Date": config_dict["Date"]
            },
            {
                "Name": "Manual",
                "Author": "Yehuda Warszawer",
                "Date": config_dict["Date"]
            },
            {
                "Name": "Manual",
                "Author": "Nathan Molinier",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }
    with open(path_json_out, 'w') as f:
        if 'label-SC_seg' in path_json_out:
            json.dump(sc_data_json, f, indent=4)
        else:
            json.dump(data_json, f, indent=4)
        print(f'Created: {path_json_out}')

if __name__=='__main__':
    main()
            
            

            