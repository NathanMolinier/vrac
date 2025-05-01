import json
import time
import argparse
import os

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Save JSON sidecars.')
    parser.add_argument('-path-json', help='Path out json sidecar. Example: derivatives/labels/sub-001/anat/sub-001_T2w_label-canal_seg.json', required=True)
    return parser

def create_json_file(path_json_out):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "totalspineseg",
                "Link": "https://github.com/neuropoly/totalspineseg",
                "Version": "r20250224",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                "Name": "nnInteractive",
                "Link": "https://github.com/MIC-DKFZ/nnInteractive",
                "Version": "PyPi 1.0.1",
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