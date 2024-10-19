from vrac.data_management.image import Image
import os
import glob
import cv2
import numpy as np
from progress.bar import Bar
from vrac.utils.utils import normalize

path_to_BIDS = '/Users/nathan/data/rootlets-ulrike/hc-leipzig-7t-mp2rage'
contrast = ''
output_folder_path = os.path.join(path_to_BIDS, 'derivatives/preview')

# Fetch all the niftii files in the BIDS folder
glob_files = glob.glob(os.path.join(path_to_BIDS,'**', f'*{contrast}.nii.gz'), recursive=True)

# Remove files from derivatives to only have images
nii_files = [file for file in glob_files if 'derivatives' not in file]

# Create output folder if it does not exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Init progression bar
bar = Bar('Convert data ', max=len(nii_files))

for file_path in nii_files:
    try:
        img = Image(file_path).change_orientation('RSP')
        arr = np.array(img.data)
        ind = arr.shape[0]//2

        # Normalize
        slice_img = arr[ind, :, :]
        percentile = np.percentile(slice_img, 95)
        slice_img_norm = normalize(np.clip(slice_img, None, percentile))

        file_name = os.path.basename(file_path).split('.')[0] + '.png'
        cv2.imwrite(os.path.join(output_folder_path, file_name), slice_img_norm*255)
    except:
        pass