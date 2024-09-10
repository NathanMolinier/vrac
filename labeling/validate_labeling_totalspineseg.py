import os
from vrac.data_management.image import Image
import numpy as np
import subprocess

img_folder = '/Users/nathan/Desktop/input'
label_folder = '/Users/nathan/Desktop/step2_output'
crop_folder = '/Users/nathan/Desktop/input_crop'
exclude_labels_top = [40, 41, 224] # Exclude  vertebrae C1, vertebrae C2 and disc C2-C3
exclude_labels_bottom = [92, 202] # Exclude sacrum, disc L5-S1

for label_name in os.listdir(label_folder):
    if label_name.endswith('.nii.gz'):
        label_path = os.path.join(label_folder, label_name)
        img_path = os.path.join(img_folder, label_name.replace('.nii.gz', '_0000.nii.gz'))
        out_path = os.path.join(crop_folder, label_name.replace('.nii.gz', '_0000_crop.nii.gz'))
        label = Image(label_path)
        img = Image(img_path)
        shape = label.data.shape

        if label.orientation == 'LPI' and img.orientation == 'LPI':
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
            
            subprocess.run([
                'sct_crop_image',
                '-i', img_path,
                '-zmin', str(z_min + 5),
                '-zmax', str(z_max - 5),
                '-o', out_path
            ])
            print(f'Image {img_path} was cropped')
        else:
            print(f'Label is {label.orientation} and image is {img.orientation}')
        