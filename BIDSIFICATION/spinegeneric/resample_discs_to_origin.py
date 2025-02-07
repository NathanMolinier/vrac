"""
This script removes unecessary use of space-other in the spinegeneric datasets.

"""

import os
import tempfile, datetime, subprocess, shutil

from vrac.data_management.image import Image
from vrac.data_management.utils import get_img_path_from_label_path

def main():
    # Add variables
    err_file = 'err.txt'

    with open(err_file, 'r') as f:
        err_list = f.readlines()

    for label_path in err_list:
        label_path = label_path.replace('\n', '')
        img_path = get_img_path_from_label_path(label_path)
        if not os.path.exists(img_path):
            raise ValueError(f'File {img_path} is missing !')
        else:
            # Copy files to a temporary folder
            tmp_dir = tempfile.mkdtemp(prefix=f"fix_discs_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
            img_dir = os.path.join(tmp_dir, 'img')
            label_dir = os.path.join(tmp_dir, 'label')
            os.makedirs(img_dir)
            os.makedirs(label_dir)

            # Load image in RPI
            img = Image(img_path).change_orientation('RPI')
            img.save(os.path.join(img_dir, os.path.basename(img_path)))

            # Load label in RPI
            label = Image(label_path).change_orientation('RPI')
            label.save(os.path.join(label_dir, os.path.basename(img_path)))

            subprocess.check_call([
                "totalspineseg_transform_seg2image",
                "-i", img_dir,
                "-s", label_dir,
                "-o", os.path.dirname(label_path),
                "--image-suffix", "",
                "--output-seg-suffix", "_label-discs_dlabel",
                "-x", "label",
                "--overwrite"
            ])

            subprocess.check_call([
                "sct_qc",
                "-i", img_path,
                "-s", label_path,
                "-p", "sct_label_vertebrae",

            ])

            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(label_path)





if __name__=='__main__':
    main()
            
            

            