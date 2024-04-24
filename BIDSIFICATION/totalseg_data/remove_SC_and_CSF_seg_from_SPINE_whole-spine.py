'''
This script extracts the spinal cord (SC) and the cerebrospinal fluid (CSF) segmentations from the spine segmentation 
to generate 2 new separate labels.

https://github.com/neuropoly/data-management/issues/308
'''

import json
import os
import numpy as np
import subprocess
import time
import glob
import shutil

from vrac.data_management.image import Image, zeros_like

def main():
    # Add variables
    bids_folder = '/Users/nathan/data/whole-spine'
    derivatives_path = os.path.join(bids_folder, 'derivatives/labels')
    qc_path = os.path.join(bids_folder, 'qc')
    SC_value = 200
    CSF_value = 201

    err = []
    missing_files = []
    for subject in os.listdir(bids_folder):
        if subject.startswith('sub'):
            for cont in ['T1w', 'T2w']:
                fname = f'{subject}_{cont}.nii.gz'
                imgs_paths = glob.glob(os.path.join(bids_folder, subject, 'anat', fname))
                if not imgs_paths:
                    print(f'File {fname} is missing !')
                    missing_files.append(fname)
                else:
                    for img_path in imgs_paths:
                        # Update fname
                        fname = os.path.basename(img_path)

                        # Init label paths
                        spine_path = os.path.join(derivatives_path, subject, 'anat', f'{fname.replace('.nii.gz', '')}_label-spine_dseg.nii.gz')
                        
                        # Init new label paths
                        csf_path = os.path.join(derivatives_path, subject, 'anat', f'{fname.replace('.nii.gz', '')}_label-CSF_seg.nii.gz')
                        
                        # Load spine label
                        spine_label = Image(spine_path).change_orientation('RPI')

                        # Check if equal size
                        img = Image(img_path).change_orientation('RPI')

                        if img.dim != spine_label.dim:
                            print(f'Size mismatch between {img_path} and {spine_path}')
                            err.append(img_path)

                        # Extract spinal canal from multi
                        if CSF_value in spine_label.data:
                            csf_label = zeros_like(spine_label)
                            csf_label.data[np.where(spine_label.data == CSF_value)] = 1
                            spine_label.data[np.where(spine_label.data == CSF_value)] = 0 # Remove CSF from spine
                            spine_label.data[np.where(spine_label.data == SC_value)] = 0 # Remove SC from spine

                            # Save canal and spine image
                            spine_label.save(spine_path)
                            csf_label.save(csf_path)

                        # Save image using RPI orientation
                        img.change_orientation('RPI').save(img_path)

                        # Copy json sidecars from spine label
                        shutil.copy(spine_path.replace('.nii.gz', '.json'), csf_path.replace('.nii.gz', '.json'))
                        
                        # QC images
                        subprocess.check_call([
                            "sct_qc", 
                            "-i", img_path,
                            "-s", csf_path,
                            "-d", csf_path,
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


if __name__=='__main__':
    main()
            
            

            