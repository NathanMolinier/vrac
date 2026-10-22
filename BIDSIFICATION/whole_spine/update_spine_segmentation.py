import json
import os

import nibabel as nib
import numpy as np

bids_dataset = '/Users/nathan/data/whole-spine'
new_segmentation_labels = '/Users/nathan/Desktop/step2_output'

name_suffix = 'sub-XXX_T?w_label-spine_dseg.nii.gz'

data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "totalspineseg",
                "Version": "r20260730",
                "Repository": "https://github.com/neuropoly/totalspineseg"
            }
        ]
    }


def main():
    derivatives_root = os.path.join(bids_dataset, 'derivatives', 'labels')

    for fname in sorted(os.listdir(new_segmentation_labels)):
        if not fname.endswith('.nii.gz'):
            continue

        base = fname[:-len('.nii.gz')]
        parts = base.split('_')
        if len(parts) != 2:
            print(f'Skipping unexpected filename: {fname}')
            continue

        sub_id, contrast = parts
        if not sub_id.startswith('sub-') or not (contrast.endswith('w') and contrast.startswith('T')):
            print(f'Skipping unexpected filename: {fname}')
            continue

        new_name = name_suffix.replace('sub-XXX', sub_id).replace('T?w', contrast)
        out_dir = os.path.join(derivatives_root, sub_id, 'anat')
        os.makedirs(out_dir, exist_ok=True)

        src_nii = os.path.join(new_segmentation_labels, fname)
        dst_nii = os.path.join(out_dir, new_name)

        img = nib.load(src_nii)
        data = np.asanyarray(img.dataobj)
        data = np.where((data == 1) | (data == 2), 0, data)
        out_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(out_img, dst_nii)

        dst_json = dst_nii[:-len('.nii.gz')] + '.json'
        with open(dst_json, 'w') as f:
            json.dump(data_json, f, indent=4)

        print(f'{src_nii} -> {dst_nii}')

    return


if __name__ == '__main__':
    main()
