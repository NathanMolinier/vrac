"""
This script removes unecessary use of space-other in the spinegeneric datasets.

"""

import os
import shutil
import glob

from vrac.data_management.image import Image
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    bids_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/data-multi-subject'
    
    for derivative in ["labels", "labels_softseg",  "labels_softseg_bin"]:
        derivatives_path = os.path.join(bids_folder, f'derivatives/{derivative}')
        
        err = []
        missing_files = []
        space_other_files = glob.glob(derivatives_path + '/**/*' + 'space-other' + '*.nii.gz', recursive=True)
        for label_path in space_other_files:
            if 'sub-amu01' in label_path:
                img_path = get_img_path_from_label_path(label_path)
                if not os.path.exists(img_path):
                    img_path = get_img_path_from_label_path(label_path.replace('_space-other', ''))
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

                    # Change image name
                    new_img_path = img_path.replace('_space-other','')
                        
                    # Rename label json sidecars
                    path_in_json = label_path.replace('.nii.gz', '.json')
                    path_out_json = new_label_path.replace('.nii.gz', '.json')
                    
                    # Rename image json sidecars
                    path_in_img_json = img_path.replace('.nii.gz', '.json')
                    path_out_img_json = new_img_path.replace('.nii.gz', '.json')

                    # Save image and replace former files
                    if new_img_path != img_path:
                        img.save(new_img_path)
                        shutil.copy(path_in_img_json, path_out_img_json)
                        print(f'Removing image NIFTII {img_path}')
                        os.remove(img_path)
                        print(f'Removing image JSON {path_in_img_json}')
                        os.remove(path_in_img_json)

                    # Save label and replace former files
                    label.save(new_label_path)
                    shutil.copy(path_in_json, path_out_json)
                    print(f'Removing label NIFTII {label_path}')
                    os.remove(label_path)
                    print(f'Removing label JSON {path_in_json}')
                    os.remove(path_in_json)
                    
                        
        print("missing files:\n" + '\n'.join(sorted(missing_files)))
        print("err files:\n" + '\n'.join(sorted(err)))

        with open('err.txt', 'w') as f:
            for line in err:
                f.write(f"{line}\n")

        with open('missing.txt', 'w') as f:
            for line in missing_files:
                f.write(f"{line}\n")



if __name__=='__main__':
    main()
            
            

            