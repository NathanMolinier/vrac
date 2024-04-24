"""
This script adds manually corrected sacrum segmentations to our internal spider dataset.

Sacrum masks are also added to spine segmentations while spinal canal segmentations are saved as a separate new labels.

https://github.com/neuropoly/data-management/issues/309
"""

import json
import os
import numpy as np
import subprocess
import time
import glob

from vrac.data_management.image import Image, zeros_like
from vrac.utils.add_mask_to_seg import add_mask_to_seg

def main():
    # Add variables
    manual_folder = '/Users/nathan/data/spider-challenge-2023/derivatives/spider/derivatives/labels'
    bids_folder = '/Users/nathan/data/spider-challenge-2023'
    derivatives_path = os.path.join(bids_folder, 'derivatives/labels')
    qc_path = os.path.join(bids_folder, 'qc')
    sacrum_value = 92
    canal_value = 201

    err = []
    missing_files = []
    for subject in os.listdir(bids_folder):
        if subject.startswith('sub'):
            for cont in ['T1w', 'T2w']:
                fname = f'{subject}_*_{cont}.nii.gz'
                imgs_paths = glob.glob(os.path.join(bids_folder, subject, 'anat', fname))
                if not imgs_paths:
                    print(f'File {fname} is missing !')
                    missing_files.append(fname)
                else:
                    for img_path in imgs_paths:
                        # Update fname
                        fname = os.path.basename(img_path)

                        # Init label paths
                        if subject in os.listdir(manual_folder):
                            sacrum_path_in = os.path.join(manual_folder, subject, 'anat', f'{subject}_{cont}_sacrum.nii.gz')
                        else:
                            sacrum_path_in = os.path.join(derivatives_path, subject, 'anat', f'{fname.replace('.nii.gz', '')}_label-sacrum_seg.nii.gz')
                        multi_path = os.path.join(derivatives_path, subject, 'anat', f'{fname.replace('.nii.gz', '')}_label-multi_dseg.nii.gz')
                        
                        # Init new label paths
                        canal_path = os.path.join(derivatives_path, subject,'anat', f'{fname.replace('.nii.gz', '')}_label-canal_seg.nii.gz')
                        spine_path = os.path.join(derivatives_path, subject,'anat', f'{fname.replace('.nii.gz', '')}_label-spine_dseg.nii.gz')
                        
                        # Load multi label
                        multi_label = Image(multi_path).change_orientation('RPI')

                        # Check if equal size
                        img = Image(img_path).change_orientation('RPI')
                        if img.dim != multi_label.dim:
                            print(f'Size mismatch between {img_path} and {multi_path}')
                            err.append(img_path)

                        # Extract spinal canal from multi
                        if canal_value in multi_label.data:
                            canal_label = zeros_like(multi_label)
                            canal_label.data[np.where(multi_label.data == canal_value)] = 1
                            multi_label.data[np.where(multi_label.data == canal_value)] = 0 # Remove canal from multi

                            # Save canal and spine image
                            multi_label.save(spine_path)
                            canal_label.save(canal_path)
                        else:
                            raise ValueError(f'Spinal canal value missing in {multi_path}')

                        # Save image using RPI orientation
                        img.change_orientation('RPI').save(img_path)

                        # Add sacrum mask to spine dseg
                        add_mask_to_seg(fname_mask=sacrum_path_in, fname_seg=spine_path, fname_file_out=spine_path, val=sacrum_value)

                        # Create or edit json sidecars
                        create_json_canal(canal_json=canal_path.replace('.nii.gz', '.json'), multi_json=multi_path.replace('.nii.gz', '.json'))
                        create_json_spine(spine_json=spine_path.replace('.nii.gz', '.json'), multi_json=multi_path.replace('.nii.gz', '.json'), sacrum_json=sacrum_path_in.replace('.nii.gz', '.json'))

                        # QC images
                        subprocess.check_call([
                            "sct_qc", 
                            "-i", img_path,
                            "-s", canal_path,
                            "-d", canal_path,
                            "-p", "sct_deepseg_lesion",
                            "-plane", "sagittal",
                            "-qc", qc_path
                        ])

                        subprocess.check_call([
                            "sct_qc", 
                            "-i", img_path,
                            "-s", spine_path,
                            "-d", spine_path,
                            "-p", "sct_deepseg_lesion",
                            "-plane", "sagittal",
                            "-qc", qc_path
                        ])
    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("err files:\n" + '\n'.join(sorted(err)))


def create_json_canal(canal_json, multi_json):
    """
    Create a json sidecar file for canal images
    :param path_file_out: path to the output file
    """
    if canal_json.endswith('.nii.gz') or multi_json.endswith('.nii.gz'):
        raise ValueError('Path to image specified ! Use .json extension !')
    
    with open(multi_json, 'r') as file:
        multi_dict = json.load(file)
        print(f'JSON {multi_json} was loaded')
    
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": multi_dict["GeneratedBy"]+[
            {
                "Name": "Manual",
                "Author": "Nathan Molinier",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }
    with open(canal_json, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Created: {canal_json}')


def create_json_spine(spine_json, multi_json, sacrum_json):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    if spine_json.endswith('.nii.gz') or multi_json.endswith('.nii.gz') or sacrum_json.endswith('.nii.gz'):
        raise ValueError('Path to image specified ! Use .json extension !')
    
    with open(multi_json, 'r') as file:
        multi_dict = json.load(file)
        print(f'JSON {multi_json} was loaded')
    
    with open(sacrum_json, 'r') as file:
        sacrum_dict = json.load(file)
        print(f'JSON {sacrum_json} was loaded')
    
    sacrum_dict["GeneratedBy"][0]["Entity"] = "Sacrum"
    sacrum_dict["GeneratedBy"][0]["Description"] = "nnUNet3D was used to add sacrum masks: see https://github.com/neuropoly/totalsegmentator-mri/issues/18"
    
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": multi_dict["GeneratedBy"]+sacrum_dict["GeneratedBy"]+[
            {
                "Name": "Manual",
                "Author": "Nathan Molinier",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }
    with open(spine_json, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Created: {spine_json}')

if __name__=='__main__':
    main()
            
            

            