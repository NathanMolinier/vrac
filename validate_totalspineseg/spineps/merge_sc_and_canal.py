from vrac.data_management.image import Image, zeros_like
import os, json, argparse
import numpy as np

def main():
    # Load variables
    with_sc_folder = "/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/totalsegXtotalspinesegXspineps/spineps/canal/pred"
    spineps_folder = "/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/totalsegXtotalspinesegXspineps/spineps/canal/pred-merged"

    for file in os.listdir(with_sc_folder):
        with_sc_path = os.path.join(with_sc_folder, file)
        img_with_sc = Image(with_sc_path).change_orientation('RPI')
        img_with_sc.data[np.where(img_with_sc.data == 60)] = 61
    
        out_path = os.path.join(spineps_folder, file)
        img_with_sc.save(out_path)

if __name__=='__main__':
    main()       
