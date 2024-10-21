import os
from vrac.data_management.image import Image
import numpy as np
import subprocess

img_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/localizer-based/raw'
label_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/localizer-based/out/step2_output'
crop_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/localizer-based/raw_crop'
exclude_labels_top = [11, 12, 63, 71, 13, 72] # Exclude  vertebrae C1, vertebrae C2, disc C2-C3 and disc C7-T1
exclude_labels_bottom = [50, 100] # Exclude sacrum, disc L5-S1

for label_name in os.listdir(img_folder):
    if label_name.endswith('.nii.gz'):
        label_path = os.path.join(label_folder, label_name)
        img_path = os.path.join(img_folder, label_name)#.replace('.nii.gz', '_0000.nii.gz'))
        out_path = os.path.join(crop_folder, label_name)
        img = Image(img_path)
        if os.path.exists(label_path):
            label = Image(label_path)

            # Keep original orientation
            ori_img = img.orientation
            ori_label = label.orientation

            # Change orientation
            img.change_orientation('LPI')
            label.change_orientation('LPI')
            shape = label.data.shape

            # Extract vertical coordinates z_min and z_max for cropping
            z_max = shape[2]-1
            for ex_label in exclude_labels_top:
                coords_label = np.where(label.data == ex_label)
                if list(coords_label[0]):
                    if np.min(coords_label[2]) < z_max:
                        z_max = np.min(coords_label[2])
            
            z_min = 0
            for ex_label in exclude_labels_bottom:
                coords_label = np.where(label.data == ex_label)
                if list(coords_label[0]):
                    if np.max(coords_label[2]) > z_min:
                        z_min = np.max(coords_label[2])
            
            # Crop image
            if not os.path.exists(crop_folder):
                os.makedirs(crop_folder)
            
            # Save img in out folder
            img.save(out_path)

            subprocess.run([
                'sct_crop_image',
                '-i', out_path,
                '-zmin', str(z_min),
                '-zmax', str(z_max),
                '-o', out_path
            ])

            subprocess.run([
                'sct_image',
                '-i', out_path,
                '-setorient', ori_img
            ])
            print(f'Image {img_path} was cropped')
        else:
            # Save img in out folder
            img.save(out_path)
        