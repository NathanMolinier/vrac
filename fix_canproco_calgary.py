from BIDSIFICATION.image import Image, zeros_like
import os
import subprocess
import cv2
import numpy as np

img_path = '/Users/nathan/Desktop/anat/sub-cal056_ses-M0_T2w.nii.gz'
sc_path = '/Users/nathan/Desktop/anat/sub-cal056_ses-M0_T2w_seg-manual.nii.gz'
out_dir = '/Users/nathan/Desktop/anat/test'
sc_name = os.path.basename(sc_path)
img_name = os.path.basename(img_path)

img = Image(img_path).change_orientation('RPI')
seg = Image(sc_path).change_orientation('RPI')

nx, ny, nz, nt, px, py, pz, pt = img.dim
snx, sny, snz, snt, spx, spy, spz, spt = seg.dim

# Resample image to real sc seg resolution
res_img_path = os.path.join(out_dir, img_name.replace('.nii.gz', '_res.nii.gz'))
subprocess.check_call(['sct_resample',
                       '-i', img_path,
                        '-mm', '0.8x0.8x0.8', 
                        '-o', res_img_path,])

# Create new segmentation with real header
res_img = Image(res_img_path).change_orientation('RPI')
new_seg = zeros_like(res_img)
new_seg.data = seg.data

# Save new segmentation
new_hd_sc_path = os.path.join(out_dir, sc_name.replace('.nii.gz', '_new-header.nii.gz'))
new_seg.save(new_hd_sc_path)

res_sc_path = os.path.join(out_dir, sc_name.replace('.nii.gz', '_res.nii.gz'))
subprocess.check_call(['sct_resample',
                       '-i', new_hd_sc_path,
                       '-x', 'linear',
                       '-mm', f'{str(px)}x{str(py)}x{str(pz)}',
                       '-o', res_sc_path,])

subprocess.check_call(['sct_maths',
                       '-i', res_sc_path,
                       '-bin', '0',
                       '-o', res_sc_path,])

res_seg = Image(res_sc_path).change_orientation('RPI')

rnx, rny, rnz, rnt, rpx, rpy, rpz, rpt = res_seg.dim

print(1)