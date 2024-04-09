import json
import time
import argparse
import os

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment an image using nnUNetV2 model.')
    parser.add_argument('-path-json', help='Input image to segment. Example: derivatives/labels/sub-001/anat/sub-001_T2w_label-sacrum_seg.json', required=True)
    return parser

def create_json_file(path_json_out, resampling):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    if os.path.exists(path_json_out):
        with open(path_json_out, 'r') as file:
            config_dict = json.load(file)
            print(f'JSON {path_json_out} was loaded')
    
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "sct_resample",
                "Version": "unknown",
                "Command": "-mm 0.8x0.8x0.8",
                "Date": config_dict["Date"]
            },
            {
                "Name": "Manual",
                "Author": config_dict["Author"],
                "Date": config_dict["Date"]
            },
            {
                "Name": "sct_resample",
                "Version": "SCT v6.1",
                "Command": f"-x linear -mm {resampling}",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                "Name": "sct_maths",
                "Version": "SCT v6.1",
                "Command": "-bin 0.5",
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

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()

    path_json_out = args.path_json

    create_json_file(path_json_out)