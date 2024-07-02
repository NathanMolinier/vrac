"""
This script fixes the format of json sidecars
"""

import json
import os


def main():
    # Add variables
    bids_folder = '/Users/nathan/data/data-spinegeneric-fix'
    
    with open(os.path.join(bids_folder, 'changed_files.txt')) as f:
        changed_files = [os.path.join(bids_folder, file.replace('\n','')) for file in f.readlines()]
    
    for path in changed_files:
        if not os.path.exists(path):
            raise ValueError(f'File {path} is missing !')
        else:
            if path.endswith('.json'):
                # Edit json sidecars
                if 'space-other' in path:
                    edit_json_file(path, orig=False)
                else:
                    edit_json_file(path, orig=True)


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
        else:
            raise ValueError(f"{len(old_json["GeneratedBy"])} dict in GeneratedBy please act")
        

    if "SpatialReference" in old_json:
        if "Other" in old_json["SpatialReference"]:
            if "root-mean square" in old_json["SpatialReference"]["Other"]:
                data_json["GeneratedBy"].append(
                    {
                        "Name": "sct_maths",
                        "Flags": "-rms",
                        "Axis": "t",
                        "Version": "SCT v6~",
                        "Date": "2024-02-20 00:00:00"
                    },
                )
        if "Resampling" in old_json["SpatialReference"]:
            data_json["GeneratedBy"].append(
                {
                    "Name": "sct_resample",
                    "Flags": "-mm",
                    "Resolution": old_json["SpatialReference"]["Resampling"],
                    "Version": "SCT v6~",
                    "Date": "2024-02-20 00:00:00"
                },
            )
        if "Reorientation" in old_json["SpatialReference"]:
            data_json["GeneratedBy"].append(
                {
                    "Name": "sct_image",
                    "Flags": "-setorient",
                    "Orientation": old_json["SpatialReference"]["Reorientation"],
                    "Version": "SCT v6~",
                    "Date": "2024-02-20 00:00:00"
                }
            )

    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Modified: {path_json_out}')

if __name__=='__main__':
    main()
            
            

            