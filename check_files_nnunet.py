"""
This script removes unecessary use of space-other in the spinegeneric datasets.

"""

import os
import glob

from vrac.data_management.image import Image
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    nnunet_labels_folder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/romane_tss_data/nnUNet/raw/Dataset99_TotalSpineSeg/labelsTr/'
    
    dim = []
    ori = []
    missing_files = []
        
    labels_to_check = glob.glob(nnunet_labels_folder + '*.nii.gz')
    for label_path in labels_to_check:
        img_path = label_path.split('_label-')[0].replace('labelsTr', 'imagesTr') + '_0000.nii.gz'
        if not os.path.exists(img_path):
            raise ValueError(f'{img_path} is missing')
        else:
            # Load image
            img = Image(img_path)

            # Load label
            label = Image(label_path)

            if img.orientation != label.orientation:
                ori.append(img_path)
                print(f'{label.orientation} for {label_path} while {img.orientation} for {img_path}')

            # Check if equal size
            if img.dim[:3] != label.dim[:3]:
                dim.append(label_path)
                print(f'ERROR with {label_path}')
                        
                        
    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("ori err files:\n" + '\n'.join(sorted(ori)))
    print("dim err files:\n" + '\n'.join(sorted(dim)))


if __name__=='__main__':
    main()
            
            

            