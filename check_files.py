"""
This script removes unecessary use of space-other in the spinegeneric datasets.

"""

import os
import glob

from vrac.data_management.image import Image
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    bids_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/data-multi-subject'
    
    err = []
    ori = []
    missing_files = []
    for derivative in ["labels", "labels_softseg",  "labels_softseg_bin"]:
        derivatives_path = os.path.join(bids_folder, f'derivatives/{derivative}')
        
        space_other_files = glob.glob(derivatives_path + '/**/*' + '*.nii.gz', recursive=True)
        for label_path in space_other_files:
            img_path = get_img_path_from_label_path(label_path)
            if not os.path.exists(img_path):
                print(f'File {img_path} is missing !')
                missing_files.append(img_path)
            else:
                # Load image in RPI
                img = Image(img_path)

                if img.orientation != 'RPI':
                    img.change_orientation('RPI')
                    img.save(img_path)
                    ori.append(img_path)
                    print(f'ORI with {img_path}')

                # Load label in RPI
                label = Image(label_path)

                if label.orientation != 'RPI':
                    label.change_orientation('RPI')
                    label.save(label_path)
                    ori.append(label_path)
                    print(f'ORI with {label_path}')

                # Check if equal size
                if img.dim[:3] != label.dim[:3]:
                    err.append(label_path)
                    print(f'ERROR with {label_path}')
                        
                        
    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("err files:\n" + '\n'.join(sorted(err)))

    with open('err.txt', 'w') as f:
        for line in err:
            f.write(f"{line}\n")

    with open('missing.txt', 'w') as f:
        for line in missing_files:
            f.write(f"{line}\n")
    
    with open('ori.txt', 'w') as f:
        for line in ori:
            f.write(f"{line}\n")



if __name__=='__main__':
    main()
            
            

            