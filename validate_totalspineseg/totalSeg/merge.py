from vrac.data_management.image import Image, zeros_like
import os, json, argparse
from progress.bar import Bar
import numpy as np

def main():
    # Load variables
    sc_folder = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-totalseg/sexy-comparison/pred_sc"
    vert_folder = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-totalseg/sexy-comparison/pred_vert"
    total_folder = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-totalseg/sexy-comparison/pred_total"

    for file in os.listdir(vert_folder):
        vert_path = os.path.join(vert_folder, file)
        sc_path = os.path.join(sc_folder, file)
        img_vert = Image(vert_path).change_orientation('RPI')
        img_sc = Image(sc_path).change_orientation('RPI')
        img_vert.data[np.where(img_sc.data == 1)] = 1
    
        out_path = os.path.join(total_folder, file)
        img_vert.save(out_path)

if __name__=='__main__':
    main()       
