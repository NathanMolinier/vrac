import numpy as np
from vrac.data_management.image import Image, zeros_like
import os
from copy import deepcopy

def main():
    benchmark_txt_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/disc-labeling-benchmark/results/files/benchmark_pediat_vanderbilt.txt'
    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/philadelphia-pediatric/sub-122/anat/sub-122_rec-composed_T1w.nii.gz'
    subject_name = 'sub-122'
    contrast = 'T1w'
    out_folder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/benchmark-example'

    with open(benchmark_txt_path, "r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
        split_lines[0][-1] = split_lines[0][-1].replace('\n','')

    txt_lines = np.array(split_lines)
    methods = txt_lines[0,:]
    methods[-1] = methods[-1].replace('\n','')

    subject_idx = np.where(methods=='subject_name')[0][0]
    discs_num_idx = np.where(methods=='num_disc')[0][0]
    contrast_idx = np.where(methods=='contrast')[0][0]

    methods_name = methods[discs_num_idx+1:]

    relevant_lines = txt_lines[txt_lines[:,subject_idx] == subject_name]
    relevant_lines = relevant_lines[relevant_lines[:,contrast_idx] == contrast]

    # open image
    img = Image(img_path).change_orientation('RIP')

    for meth in methods_name:
        method_idx = np.where(methods==meth)[0][0]

        # Create segmentation
        seg = zeros_like(img)
        RL_coord = seg.data.shape[0]//2

        # Constucting 
        print(f"Constructing {meth}")
        for line in relevant_lines:
            value = int(line[discs_num_idx])
            coords_2d = str(line[method_idx]).replace('[','').replace(']','').replace('\n','').split(',')
            if not coords_2d[0] == 'None':
                coords_2d[0], coords_2d[1] = int(coords_2d[0]), int(coords_2d[1])
                seg.data[RL_coord, coords_2d[0], coords_2d[1]] = value
        
        # Save seg
        seg.change_orientation('LPI').save(os.path.join(out_folder, 'seg', str(meth) + '.nii.gz'))
        deepcopy(img).change_orientation('LPI').save(os.path.join(out_folder, 'img', str(meth) + '.nii.gz'))

if __name__=='__main__':
    main()