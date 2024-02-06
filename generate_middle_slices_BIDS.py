from BIDSIFICATION.image import Image
import os
import glob
import cv2
import numpy as np
from progress.bar import Bar

path_to_BIDS = '/Users/nathan/data/whole-spine'
output_folder_path = os.path.join(path_to_BIDS, 'derivatives/preview')

# Fetch all the niftii files in the BIDS folder
glob_files = glob.glob(os.path.join(path_to_BIDS,'**', '*.nii.gz'), recursive=True)

# Remove files from derivatives to only have images
nii_files = [file for file in glob_files if 'derivatives' not in file]

# Create output folder if it does not exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Init progression bar
bar = Bar('Convert data ', max=len(nii_files))

for file_path in nii_files:
    img = Image(file_path).change_orientation('RSP')
    arr = np.array(img.data)
    ind = arr.shape[0]//2

    file_name = os.path.basename(file_path).split('.')[0] + '.png'
    cv2.imwrite(os.path.join(output_folder_path, file_name), arr[ind, :, :])