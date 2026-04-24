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
        "C1":11,
        "C2":12,
        "C3":13,
        "C4":14,
        "C5":15,
        "C6":16,
        "C7":17,
        "T1":21,
        "T2":22,
        "T3":23,
        "T4":24,
        "T5":25,
        "T6":26,
        "T7":27,
        "T8":28,
        "T9":29,
        "T10":30,
        "T11":31,
        "T12":32,
        "L1":41,
        "L2":42,
        "L3":43,
        "L4":44,
        "L5":45,
        "S1":50
    }

    # Load variables
    folder = os.path.abspath(args.folder)
    out_path = os.path.abspath(args.out_path)
    
    # Pre-filter vertebrae files and sort for consistent processing
    vertebrae_files = []
    for file in os.listdir(folder):
        if file.startswith('vertebrae'):
            vert = file.split('_')[-1].replace('.nii.gz', '')
            if vert in vert_dict:
                vertebrae_files.append((file, vert, vert_dict[vert]))
    
    if not vertebrae_files:
        print("No vertebrae files found")
        return
    
    # Load first image to initialize output array
    first_path = os.path.join(folder, vertebrae_files[0][0])
    first_img = Image(first_path)
    out_img = zeros_like(first_img)
    
    # Process first file
    out_img.data[first_img.data == 1] = vertebrae_files[0][2]
    
    # Process remaining files
    for file, vert, vert_id in vertebrae_files[1:]:
        in_path = os.path.join(folder, file)
        img = Image(in_path)
        # Direct boolean indexing is faster than np.where()
        out_img.data[img.data == 1] = vert_id
    
    out_img.save(out_path)

if __name__=='__main__':
    main()       
