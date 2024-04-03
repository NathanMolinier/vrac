from BIDSIFICATION.image import Image, zeros_like
from utils.utils import tmp_create

import os
import subprocess
import numpy as np
import shutil

bids_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-canproco'
derivatives_path = 'derivatives/labels'
resample = True

missing_files = []
for sub in os.listdir(bids_path):
    for ses in ['ses-M0', 'ses-M12']:
        if 'cal' in sub:
            img_path = os.path.join(bids_path, sub, f'{ses}/anat/', f'{sub}_{ses}_T2w.nii.gz')
            sc_path = os.path.join(bids_path, derivatives_path, sub, 'ses-M0/anat/', f'{sub}_ses-M0_T2w_seg-manual.nii.gz')
            if not os.path.exists(img_path) or not os.path.exists(sc_path):
                missing_files.append(img_path)
            else:
                # Load image and SC seg
                img = Image(img_path).change_orientation('RPI')
                seg = Image(sc_path).change_orientation('RPI')

                nx, ny, nz, nt, px, py, pz, pt = img.dim
                snx, sny, snz, snt, spx, spy, spz, spt = seg.dim

                if sny == 320 and snz == 320 and ny == 512 and nz == 512 and py == spy and pz == spz:
                    print(f'Fixing subject {sub}')
                    
                    # Save image to RPI orientation
                    img.save(img_path)

                    # Create tempdirectory
                    tmpdir = tmp_create(basename='canproco-fix')

                    # Fetch basename
                    sc_name = os.path.basename(sc_path)
                    img_name = os.path.basename(img_path)

                    # Resample image to real sc seg resolution
                    res_img_path = os.path.join(tmpdir, img_name.replace('.nii.gz', '_res.nii.gz'))
                    subprocess.check_call(['sct_resample',
                                        '-i', img_path,
                                        '-mm', '0.8x0.8x0.8', 
                                        '-o', res_img_path])

                    # Create new segmentation from original header
                    res_img = Image(res_img_path).change_orientation('RPI') # Change orientation to RPI
                    new_seg = zeros_like(res_img)
                    new_seg.data = seg.data

                    # Overwrite old segmentation
                    new_seg.save(sc_path)
                    
                    # Resample to same resolution as the image
                    if resample:
                        # Resample to original image resolution with linear interpolation
                        subprocess.check_call(['sct_resample',
                                            '-i', sc_path,
                                            '-x', 'linear',
                                            '-mm', f'{str(px)}x{str(py)}x{str(pz)}',
                                            '-o', sc_path,])
                        
                        # Binarize mask
                        subprocess.check_call(['sct_maths',
                                            '-i', sc_path,
                                            '-bin', '0.5',
                                            '-o', sc_path,])

                    # QC segmentations
                    subprocess.check_call(['sct_qc',
                                            '-i', img_path,
                                            '-s', sc_path,
                                            '-p', 'sct_deepseg_sc',
                                            '-qc', os.path.join(bids_path, 'qc'),])

                    # Remove tempdir
                    shutil.rmtree(tmpdir)
print("missing files:\n" + '\n'.join(sorted(missing_files)))