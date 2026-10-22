import json
import os
import shutil

bids_dataset = '/Users/nathan/data/whole-spine'
new_discs_labels = '/Users/nathan/Desktop/step1_levels'

name_suffix = 'sub-XXX_T?w_label-discs_dlabel.nii.gz'

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

    for fname in sorted(os.listdir(new_discs_labels)):
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

        src_nii = os.path.join(new_discs_labels, fname)
        dst_nii = os.path.join(out_dir, new_name)
        shutil.copy2(src_nii, dst_nii)

        dst_json = dst_nii[:-len('.nii.gz')] + '.json'
        with open(dst_json, 'w') as f:
            json.dump(data_json, f, indent=4)

        print(f'{src_nii} -> {dst_nii}')

    return


if __name__ == '__main__':
    main()
