"""
This script removes unecessary use of space-other in the spinegeneric datasets.

"""

import os
import glob

from vrac.data_management.image import Image
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    bids_folder = '/Users/nathan/data/data-multi-subject'
    key_word = 'canal'
    
    dim = []
    ori = []
    missing_files = []
        
    labels_to_check = glob.glob(bids_folder + '/derivatives/labels' + '/**/' + f'*{key_word}*.nii.gz', recursive=True)
    for label_path in labels_to_check:
        img_path = get_img_path_from_label_path(label_path)
        if not os.path.exists(img_path):
            print(f'File {img_path} is missing !')
            missing_files.append(img_path)
        else:
            # Load image
            img = Image(img_path)

            # Load label
            label = Image(label_path)

            if img.orientation != label.orientation:
                ori.append(img_path)
                print(f'{label.orientation} for {label_path} while {img.orientation} for {img_path}')

            # Check if equal size
            img.change_orientation('RPI')
            label.change_orientation('RPI')
            if img.dim[:3] != label.dim[:3]:
                dim.append(label_path)
                print(f'ERROR with {label_path}')
                        
                        
    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("ori err files:\n" + '\n'.join(sorted(ori)))
    print("dim err files:\n" + '\n'.join(sorted(dim)))


if __name__=='__main__':
    main()
            
            

            