import os
import numpy as np
from vrac.data_management.image import Image, zeros_like

def main():
    in_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/canal-eval/out/step2_output'
    out_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/canal-eval/pred'

    for fname in os.listdir(in_folder):
        path_in = os.path.join(in_folder, fname) 
        path_out = os.path.join(out_folder, fname)
        structure_labels = [1, 2]

        # Load image
        img = Image(path_in).change_orientation('RSP')
        out_img = zeros_like(img)

        # Map structure
        out_img.data = np.isin(img.data, structure_labels)

        # Save out image
        out_img.save(path_out)

if __name__=='__main__':
    main()