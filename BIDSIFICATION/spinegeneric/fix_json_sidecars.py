"""
This script fixes the format of json sidecars
"""

import json
import os

from vrac.data_management.image import Image
from vrac.data_management.utils import get_img_path_from_label_path
import glob

def main():
    # Add variables
    bids_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/data-multi-subject'

    for derivative in ["labels", "labels_softseg",  "labels_softseg_bin"]:
        derivatives_path = os.path.join(bids_folder, f'derivatives/{derivative}')
        
        err = []
        json_files = glob.glob(derivatives_path + '/**/*' + '*.json', recursive=True)
    
        for path in json_files:
            label_path = path.replace('.json', '.nii.gz')

            img_path = get_img_path_from_label_path(label_path)

            # Load image in RPI
            img = Image(img_path).change_orientation('RPI')

            # Load label in RPI
            label = Image(label_path).change_orientation('RPI')

            # Check if equal size
            if img.dim[:3] != label.dim[:3]:
                print(f"Size mismatch with label {label_path}")
                err.append(label_path)
            else:
                # Edit json sidecars
                edit_json_file(path, orig=True)
    
    with open('err.txt', 'w') as f:
        for line in err:
            f.write(f"{line}\n")


def edit_json_file(path_json_out, orig):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    if path_json_out.endswith('.nii.gz'):
        raise ValueError('Path to image specified ! Use .json extension !')
    
    if os.path.exists(path_json_out):
        with open(path_json_out, 'r') as file:
            old_json = json.load(file)
            print(f'JSON {path_json_out} was loaded')
    
    data_json = {
        "SpatialReference": "orig" if orig else old_json["SpatialReference"],
        "GeneratedBy": []
    }

    if "Author" in old_json or "author" in old_json:
        if "label-SC_seg" in path_json_out:
            data_json["GeneratedBy"].append(
                {
                    "Name": "sct_deepseg_sc",
                    "Version": "SCT v6.2"
                }
            )
        data_json["GeneratedBy"].append(
            {
                "Name": "Manual",
                "Author": old_json["Author"] if "Author" in old_json else old_json["author"],
                "Date": old_json["Date"] if "Date" in old_json else "YYYY-MM-DD 00:00:00"
            }
        )
    
    if "GeneratedBy" in old_json:
        if len(old_json["GeneratedBy"]) == 1:
            if "Manual label of C2 and C5" in old_json["GeneratedBy"][0]["Description"]:
                data_json["GeneratedBy"].append(
                {
                    "Name": "Manual",
                    "Author": "Paul Bautin",
                    "Date": "2020-07-30 11:57:36",
                    "Description": old_json["GeneratedBy"][0]["Description"]
                }
                )
            elif "Created from label-CSF_seg" in old_json["GeneratedBy"][0]["Description"]:
                data_json["GeneratedBy"].append(
                {
                    "Name": "Manual",
                    "Description": old_json["GeneratedBy"][0]["Description"]
                }
                )
            elif "Warp from T2w" in old_json["GeneratedBy"][0]["Description"]:
                data_json["GeneratedBy"].append(old_json["GeneratedBy"][0])
            else:
                raise ValueError(f"Unknow json for file {path_json_out}")
        else:
            raise ValueError(f"{len(old_json["GeneratedBy"])} dict in GeneratedBy please act")

    print("old", old_json)
    print("new", data_json)
    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Modified: {path_json_out}')

if __name__=='__main__':
    main()
            
            

            