from vrac.data_management.image import Image, zeros_like
import os, json, argparse
from progress.bar import Bar
import numpy as np

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Merge totalSeg masks.')
    parser.add_argument('--folder', '-f', required=True, help='Path to the folder containing the masks (Required)')
    parser.add_argument('--out-path', '-o', required=True, help='Path to the output file (Required)')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    vert_dict = {
        "T12":32,
        "L1":41,
        "L2":42,
        "L3":43,
        "L4":44,
        "L5":45,
    }

    # Load variables
    folder = os.path.abspath(args.folder)
    out_path = os.path.abspath(args.out_path)
    shape = None

    for file in os.listdir(folder):
        vert = file.split('_')[-1].replace('.nii.gz', '')
        if vert in vert_dict.keys():
            in_path = os.path.join(folder, file)
            img = Image(in_path)
            if shape is None:
                shape = img.data.shape
                out_img = zeros_like(img)
            out_img.data[np.where(img.data == 1)] = vert_dict[vert]
    
    out_img.save(out_path)

if __name__=='__main__':
    main()       
