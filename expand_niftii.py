from vrac.data_management.image import Image, zeros_like
import os
import numpy as np


def main():
    in_folder = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/fast-mri/split"
    out_folder = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/fast-mri/split-expand2"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for file in os.listdir(in_folder):
        file_path = os.path.join(in_folder, file)
        
        # Load image
        img = Image(file_path).change_orientation('RPI')

        # Replicate slice in the R-L direction
        img_out = zeros_like(img)
        img_out.data = np.concatenate([img.data]*2, axis=0)

        # Save output
        out_path = os.path.join(out_folder, os.path.basename(file_path))
        img_out.save(out_path)


if __name__=='__main__':
    main()