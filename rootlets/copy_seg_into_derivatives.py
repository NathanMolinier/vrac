import os
import shutil
import json
import time

def main():
    seg_folder = '/Users/nathan/Desktop/test-nninteractive/prediction_nnUNetTrainer50'
    derivatives_folder = '/Users/nathan/data/lbp-lumbar-usf-2024/derivatives/labels'

    for file in os.listdir(seg_folder):
        if file.endswith('_T2w.nii.gz'):
            subject_id = file.split('_')[0]
            ses_id = file.split('_')[1]
            seg_path = os.path.join(seg_folder, file)
            out_seg_path = os.path.join(derivatives_folder, subject_id, ses_id, 'anat', f'{subject_id}_{ses_id}_acq-isotropic_T2w_label-rootlets_seg.nii.gz')
            os.makedirs(os.path.dirname(out_seg_path), exist_ok=True)
            shutil.copy(seg_path, out_seg_path)
            print(f'Copied: {seg_path} -> {out_seg_path}')
            json_path = out_seg_path.replace('.nii.gz', '.json')
            add_json_sidecar(json_path)

def add_json_sidecar(json_path):
    json_content = {
    "SpatialReference": "orig",
    "GeneratedBy": [
        {
            "Name": "Active Learning",
            "Author": "Nathan Molinier",
            "Date": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    ]
}
    with open(json_path, 'w') as f:
        json.dump(json_content, f, indent=4)
    print(f'Created JSON sidecar: {json_path}')

if __name__ == "__main__":
    main()