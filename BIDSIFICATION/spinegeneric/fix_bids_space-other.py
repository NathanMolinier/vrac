"""
This script removes unecessary use of space-other in the spinegeneric datasets.

"""

import json
import os
import numpy as np
import subprocess
import shutil
import time
from copy import copy
import glob

from vrac.data_management.image import Image, zeros_like
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    bids_folder = '/Users/nathan/data/data-spinegeneric-fix'
    derivatives_path = os.path.join(bids_folder, 'derivatives/labels')
    
    err = []
    missing_files = []
    space_other_files = glob.glob(derivatives_path + '/**/*' + 'space-other' + '*.nii.gz', recursive=True)
    for label_path in space_other_files:
        img_path = get_img_path_from_label_path(label_path.replace('_space-other',''))
        if not os.path.exists(img_path):
            print(f'File {img_path} is missing !')
            missing_files.append(img_path)
        else:
            
            # Load image in RPI
            img = Image(img_path).change_orientation('RPI')

            # Load label in RPI
            label = Image(label_path).change_orientation('RPI')

            # Check if equal size
            if img.dim[:3] != label.dim[:3]:
                err.append(label_path)
                # Move existing entity space-other after image suffix
                dirname = os.path.dirname(label_path)
                filename = os.path.basename(label_path)
                new_filename_split = filename.replace('_space-other', '').split('_')
                suffix_pos = [len(part.split('-')) for part in new_filename_split].index(1)
                new_filename_split.insert(suffix_pos + 1, 'space-other')
                new_label_path = os.path.join(dirname, "_".join(new_filename_split))
            else:
                new_label_path = label_path.replace('_space-other', '')

            # Copy and edit json sidecars
            path_in_json = label_path.replace('.nii.gz', '.json')
            path_out_json = new_label_path.replace('.nii.gz', '.json')
            shutil.copy(path_in_json, path_out_json)

            # Save files using RPI orientation
            img.save(img_path)
            label.save(new_label_path)

            # Remove files
            print(f'Removing NIFTII {label_path}')
            os.remove(label_path)
            print(f'Removing JSON {path_in_json}')
            os.remove(path_in_json)
                    
    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("err files:\n" + '\n'.join(sorted(err)))


if __name__=='__main__':
    main()
            
            

            