from vrac.data_management.image import Image
from vrac.utils.utils import normalize

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy

dataset_path = '/Users/nathan/data/lumbar-temp'

sag_images_paths = [path for path in glob.glob(dataset_path + '/**/*' + 'acq-sag' + '*.nii.gz', recursive=True) if 'derivatives' not in path]

for path in sag_images_paths:
    # Load image
    img = Image(path).change_orientation('RSP')

    # Extract image slice
    shape = img.data.shape
    img_slice = img.data[shape[0]//2]

    # Create image without maximum
    img_slice_noMax = copy.copy(img_slice[:,:])
    percentile = np.percentile(img_slice_noMax, 95)
    img_slice_noMax = np.clip(img_slice_noMax, None, percentile)

    # Normalize and flatten data
    norm_data = normalize(img_slice)
    norm_data_noMax = normalize(img_slice_noMax)
    flat_data = norm_data.flatten()
    flat_data_noMax = norm_data_noMax.flatten()

    # Save 2d slice
    cv2.imwrite('img.png', norm_data*255)
    cv2.imwrite('imgNoMax.png', norm_data_noMax*255)

    # Plot histogram
    plt.figure()
    plt.hist(flat_data, 255)
    plt.xlim([0,1])
    plt.savefig('test.png')

    plt.figure()
    plt.hist(flat_data_noMax, 255)
    plt.xlim([0,1])
    plt.savefig('test_noMax.png')
