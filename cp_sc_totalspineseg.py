import os
from vrac.data_management.image import Image, zeros_like
import numpy as np
import json
import time
import subprocess

def create_json_file(path_json_out):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "TotalSpineSeg",
                "Version": "r20241005",
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


def main():
    folder_path = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/violin-plots/out/step1_cord'
    out_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/lumbar-vanderbilt/derivatives/labels'
    img_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/lumbar-vanderbilt'

    for file_name in os.listdir(folder_path):
        path = os.path.join(folder_path,file_name)
        sub_name = file_name.split('_')[0]
        img_path = os.path.join(img_folder, sub_name, 'anat', file_name)

        # Load image
        seg = Image(path)
        out = zeros_like(seg)

        # Binarize sc seg
        out.data = np.where(seg.data>0.5, 1, 0)

        # Name output
        out_name = file_name.replace('.nii.gz', '_label-SC_seg.nii.gz')
        out_path = os.path.join(out_folder, sub_name, 'anat', out_name)

        # Save sc segmentation
        out.save(out_path)
        create_json_file(path_json_out=out_path.replace('.nii.gz', '.json'))

        # QC using SCT
        subprocess.check_call(['sct_qc',
                                '-i', img_path,
                                '-s', out_path,
                                '-p', 'sct_deepseg_sc',
                                '-qc', os.path.join('/home/GRAMES.POLYMTL.CA/p118739/data/datasets/lumbar-vanderbilt', 'qc')])

if __name__=='__main__':
    main()


