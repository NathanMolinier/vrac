"""
The aim of this script is to use spineps and totalspineseg predictions to correct vertebrae and discs labels for the spider-challenge-2023 dataset.
"""

import argparse, os, glob, json, shutil
from vrac.data_management.image import Image
import time 
import numpy as np

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Refine segmentation using nnInteractive.')
    parser.add_argument('-spineps', required=True, help='Path to the predictions of spineps (Required)')
    parser.add_argument('-totalspineseg', required=True, help='Path to the predictions of totalspineseg (Required)')
    parser.add_argument('-labels', required=True, help='Path to the ground truth labels to correct (Required)')
    parser.add_argument('-ofolder', required=True, help='Output folder (Required)')
    return parser

def update_json_file(path_json_out):
    """
    Update json sidecar file after labeling correction
    :param path_file_out: path to the output file
    """

    with open(path_json_out, 'r') as file:
        data_json = json.load(file)
        print(f'JSON {path_json_out} was loaded')

    if not "SpatialReference" in data_json.keys():
        data_json['SpatialReference'] = 'orig'
    
    data_json['GeneratedBy'].append(
            {
                "Name": "spineps",
                "Link": "https://github.com/Hendrik-code/spineps",
                "Commit": "3deb01253fe9dbf369c8c11e21b6d78eae93417c",
                "Description": "Correction of the labels only",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    )
    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Created: {path_json_out}')

def main():
    # Load parser
    parser = get_parser()
    args = parser.parse_args()

    # Define paths
    spineps_folder = args.spineps
    tss_folder = args.totalspineseg
    labels_folder = args.labels
    output_folder = args.ofolder

    # List paths
    labels_files = glob.glob(labels_folder + "/**/*_T2w_label-spine_dseg.nii.gz", recursive=True)
    problem_spineps_list = ['sub-073', 'sub-058', 'sub-005', 'sub-253', 'sub-207', 'sub-082', 'sub-245', 'sub-015', 'sub-129', 'sub-001', 'sub-161', 'sub-110']
    problem_tss_list = ['sub-107', 'sub-085']
    problem_seg_list = ['sub-047']
    updated_list = []
    for labels_file in labels_files:
        # Fetch subject
        sub_acq = os.path.basename(labels_file).split('_T2w')[0]

        if sub_acq.split('_')[0] not in problem_spineps_list:

            # Fetch spineps segmentation
            gl = glob.glob(os.path.join(spineps_folder, sub_acq + "*_seg-vert_msk.nii.gz"))
            if len(gl) > 1:
                raise ValueError(f'Multiple files detected for {sub_acq}: {"\n".join(gl)}')
            
            spineps_file = gl[0]
            
            gl = glob.glob(os.path.join(tss_folder, sub_acq + "_T2w.nii.gz"))
            if len(gl) > 1:
                raise ValueError(f'Multiple files detected for {sub_acq}: {"\n".join(gl)}')
            
            tss_file = gl[0]

            # Load images
            labels_image = Image(labels_file).change_orientation('RPI')
            spineps_image = Image(spineps_file).change_orientation('RPI')
            tss_image = Image(tss_file).change_orientation('RPI')

            # Check unique values in labels_image
            labels_unique = [val for val in np.unique(labels_image.data) if val != 0]

            # Check if T1w image exists
            labels_image_t1w = None
            if os.path.exists(labels_file.replace('T2w', 'T1w')):
                labels_file_t1w = labels_file.replace('T2w', 'T1w')
                labels_image_t1w = Image(labels_file_t1w).change_orientation('RPI')
                labels_unique_t1w = [val for val in np.unique(labels_image.data) if val != 0]
                if not sorted(labels_unique_t1w) == sorted(labels_unique):
                    raise ValueError(f'Unique values are different between T1w and T2w for {labels_file}')

            # Loop over the values and find corresponding structures for spineps and totalspineseg
            new_val_list = []
            spineps_list = []
            new_val_pos = []
            tss_list = []
            for val in labels_unique: # Loop starts from the bootm lumbar vertebrae
                mask_struc = np.where(labels_image.data == val, 1, 0).astype(bool)
                spineps_val = np.median(spineps_image.data[mask_struc]) # Because segmentations are good, dice > 0.5, the median is used
                tss_val = np.median(tss_image.data[mask_struc])
                z_pos = np.mean(np.where(labels_image.data == val)[2])

                # Corresponding structure
                label_struc = rev_old_mapping[val]
                if not sub_acq.split('_')[0] in problem_tss_list:
                    tss_struc = rev_new_mapping[tss_val]
                else:
                    if 'L6' in new_val_list and rev_spineps_mapping[spineps_val] == 'L5-S':
                        tss_struc = 'L5-L6'
                    elif 'L6' in new_val_list and rev_spineps_mapping[spineps_val] == 'L6-S':
                        tss_struc = 'L5-S'
                    else:
                        tss_struc = rev_spineps_mapping[spineps_val]
                    
                if sub_acq.split('_')[0] in problem_spineps_list: # Copy tss classes in problematic cases
                    spineps_struc = tss_struc
                elif spineps_val == 0 and tss_val == 31 and sub_acq.split('_')[0] == 'sub-097':
                    spineps_struc = rev_spineps_mapping[19]
                elif spineps_list and rev_spineps_mapping[spineps_val] == spineps_list[-1] and not '-' in rev_spineps_mapping[spineps_val]:
                    spineps_struc = rev_spineps_mapping[spineps_val-1]
                else:
                    spineps_struc = rev_spineps_mapping[spineps_val]

                if spineps_list and spineps_struc == "error": # Spineps errors only occurs on discs for now
                    prev_struc = spineps_list[-1]
                    upper_vert = prev_struc.split('-')[0]
                    if upper_vert == 'L1':
                        spineps_struc = 'T12-L1'
                    else:
                        spineps_struc = f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}'

                if spineps_struc == tss_struc and not tss_struc in new_val_list:
                    if 'L6' in spineps_list and not 'L6' in tss_list and spineps_struc != 'sacrum' and not 'sacrum' in tss_list: # Deal with L6 + 11 thoracic spineps
                        if not '-' in tss_struc: # Vertebrae
                            struc = f'T{int(tss_struc.split('T')[-1])+1}'
                        else: # Discs
                            if tss_struc == 'T11-T12':
                                struc = 'T12-L1'
                            else:
                                upper_vert = tss_struc.split('-')[0]
                                struc = f'T{int(upper_vert.split('T')[-1])+1}-T{int(upper_vert.split('T')[-1])+2}'
                        new_val = new_mapping[struc]
                    else:
                        # Update segmentation value
                        new_val = new_mapping[tss_struc]
                else:
                    if spineps_struc == 'T13' and tss_struc == 'T12':
                        new_val = new_mapping['T12']
                    elif 'T13' in spineps_list and not 'L6' in spineps_list and 'L5' in spineps_list:
                        if spineps_struc == 'T12' and tss_struc == 'T11':
                            new_val = new_mapping['T11']
                        elif spineps_struc == 'T12-L1' and tss_struc == 'T11-T12': # T12-T13, T13-L1 and T12-L1 have same value for spineps 
                            new_val = new_mapping['T11-T12']
                        else:
                            if not '-' in tss_struc: # Vertebrae
                                if tss_struc == f'T{int(spineps_struc.split('T')[-1])-1}':
                                    new_val = new_mapping[tss_struc]
                                else:
                                    raise ValueError('New mismatch, please act')
                            else:
                                upper_vert = spineps_struc.split('-')[0]
                                if f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}' == tss_struc:
                                    new_val = new_mapping[tss_struc]
                                else:
                                    raise ValueError('New mismatch, please act')
                    elif spineps_struc.startswith('L'):
                        if 'L5' not in new_val_list and 'L4' in new_val_list and spineps_struc == 'L4-L5': # Assign value 100 to last disc before sacrum when 4 lumbar
                            new_val = new_mapping['L5-S']
                        elif 'L6' in new_val_list and '-' in spineps_struc:
                            if spineps_struc == 'L6-S':
                                new_val = new_mapping['L5-S']
                            elif spineps_struc == 'L5-S':
                                new_val = new_mapping['L5-L6']
                            else:
                                new_val = new_mapping[spineps_struc]
                        else:
                            new_val = new_mapping[spineps_struc]
                        if sub_acq not in updated_list:
                            updated_list.append(sub_acq)
                    elif 'L6' in spineps_list:
                        if spineps_struc == 'T13' and tss_struc == 'T11':
                            new_val = new_mapping['T12']
                        elif 'T13' in spineps_list:
                            if not '-' in tss_struc: # Vertebrae
                                if f'T{int(spineps_struc.split('T')[-1])-2}' == tss_struc:
                                    new_val = new_mapping[f'T{int(spineps_struc.split('T')[-1])-1}']
                                elif f'T{int(spineps_struc.split('T')[-1])-1}' == tss_struc:
                                    new_val = new_mapping[tss_struc]
                                else:
                                    raise ValueError('New mismatch, please act')
                            else: # Discs
                                if spineps_struc == 'T12-L1' and tss_struc == 'T11-T12' and not 'L6' in tss_list:
                                    new_val = new_mapping[spineps_struc]
                                elif spineps_struc == 'T12-L1' and tss_struc == 'T10-T11' and not 'L6' in tss_list:
                                    new_val = new_mapping['T11-T12']
                                elif spineps_struc == 'T12-L1' and tss_struc == 'T11-T12' and 'L6' in tss_list:
                                    new_val = new_mapping['T11-T12']
                                else:
                                    upper_vert = spineps_struc.split('-')[0]
                                    if f'T{int(upper_vert.split('T')[-1])-2}-T{int(upper_vert.split('T')[-1])-1}' == tss_struc:
                                        new_val = new_mapping[f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}']
                                    elif f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}' == tss_struc:
                                        new_val = new_mapping[tss_struc]
                                    else:
                                        raise ValueError('New mismatch, please act')
                        else:
                            if not '-' in tss_struc: # Vertebrae
                                if f'T{int(spineps_struc.split('T')[-1])-1}' == tss_struc:
                                    new_val = new_mapping[spineps_struc]
                                else:
                                    if 'T12' not in spineps_list: # Deal with spineps detecting patients with only 11 thoracic vertebrae
                                        if f'T{int(tss_struc.split('T')[-1])-1}' == spineps_struc:
                                            new_val = new_mapping[tss_struc]
                                        else:
                                            raise ValueError('New mismatch, please act')
                                    else:
                                        raise ValueError('New mismatch, please act')
                            else:
                                if spineps_struc == 'T12-L1' and tss_struc == 'T11-T12':
                                    new_val = new_mapping[spineps_struc]
                                else:
                                    upper_vert = spineps_struc.split('-')[0]
                                    if f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}' == tss_struc:
                                        new_val = new_mapping[spineps_struc]
                                    else:
                                        if 'T12' not in spineps_list: # Deal with spineps detecting patients with only 11 thoracic vertebrae
                                            upper_vert = tss_struc.split('-')[0]
                                            if spineps_struc == 'T11-T12' and tss_struc == 'T12-L1':
                                                new_val = new_mapping[tss_struc]
                                            elif f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}' == spineps_struc:
                                                new_val = new_mapping[tss_struc]
                                            else:
                                                raise ValueError('New mismatch, please act')
                                        else:
                                            raise ValueError('New mismatch, please act')
                    elif 'L5' not in new_val_list and 'L4' in new_val_list: # Deal with 4 lumbar vertebrae
                        if sub_acq not in updated_list:
                            raise ValueError(f'{sub_acq} should be in updated_list')
                        if tss_struc == 'L1-L2' and (spineps_struc == 'T12-L1' or spineps_struc == 'T11-T12'):
                            new_val = new_mapping['T12-L1']
                        elif tss_struc == 'L1' and (spineps_struc == 'T11' or spineps_struc == 'T12' or spineps_struc == 'T13'):
                            new_val = new_mapping['T12']
                        elif 'T13' in spineps_list and spineps_struc == tss_struc and tss_struc in new_val_list:
                            if not '-' in tss_struc: # Vertebrae
                                struc = f'T{int(spineps_struc.split('T')[-1])-1}'
                            else: # Discs
                                upper_vert = spineps_struc.split('-')[0]
                                struc = f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}'
                            new_val = new_mapping[struc]
                        elif 'T13' not in spineps_list:
                            if not '-' in spineps_struc: # Vertebrae
                                if f'T{int(tss_struc.split('T')[-1])-1}' == spineps_struc: 
                                    new_val = new_mapping[spineps_struc]
                                elif f'T{int(tss_struc.split('T')[-1])-2}' == spineps_struc: # 11 thoracic
                                    new_val = new_mapping[f'T{int(tss_struc.split('T')[-1])-1}']
                                else:
                                    raise ValueError('New mismatch, please act')
                            else: # Discs
                                upper_vert = tss_struc.split('-')[0]
                                if f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}' == spineps_struc:
                                    new_val = new_mapping[spineps_struc]
                                elif f'T{int(upper_vert.split('T')[-1])-2}-T{int(upper_vert.split('T')[-1])-1}' == spineps_struc: # 11 thoracic
                                    new_val = new_mapping[f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}']
                                else:
                                    raise ValueError('New mismatch, please act')
                        else:
                            raise ValueError('New mismatch, please act')
                    elif spineps_struc in new_val_list and tss_struc not in new_val_list and f'T{int(spineps_struc.split('T')[-1])-1}' == tss_struc: # Deal with spineps detecting multiple time the same vertebrae
                        new_val = new_mapping[tss_struc]
                    elif 'L6' in tss_list and 'L6' not in spineps_list:
                        new_val = new_mapping[spineps_struc]
                    elif 'T12' not in spineps_list: # Deal with spineps detecting patients with only 11 thoracic vertebrae
                        if not '-' in spineps_struc: # vertebrae
                            if f'T{int(tss_struc.split('T')[-1])-1}' == spineps_struc:
                                new_val = new_mapping[tss_struc]
                            else:
                                raise ValueError('New mismatch, please act')
                        else:
                            upper_vert = tss_struc.split('-')[0]
                            if spineps_struc == 'T11-T12' and tss_struc == 'T12-L1':
                                new_val = new_mapping[tss_struc]
                            elif f'T{int(upper_vert.split('T')[-1])-1}-T{int(upper_vert.split('T')[-1])}' == spineps_struc:
                                new_val = new_mapping[tss_struc]
                            else:
                                raise ValueError('New mismatch, please act')
                    elif spineps_struc == "error":
                        new_val = new_mapping[tss_struc]
                    else:
                        if spineps_struc == tss_struc and sub_acq.split('_')[0] in problem_seg_list:
                            new_val = new_mapping[tss_struc]
                        else:
                            raise ValueError('New mismatch, please act')
                
                # Check if val not already assigned
                if rev_new_mapping[new_val] in new_val_list and sub_acq.split('_')[0] not in problem_seg_list:
                    raise ValueError(f'Val {rev_new_mapping[new_val]} was already assigned !')
                elif rev_new_mapping[new_val] in new_val_list and sub_acq.split('_')[0] in problem_seg_list:
                    pass
                else:
                    new_val_list.append(rev_new_mapping[new_val])
                    new_val_pos.append(z_pos)
                    spineps_list.append(spineps_struc)
                    tss_list.append(tss_struc)
                
                labels_image.data[mask_struc] = new_val
                if not labels_image_t1w is None:
                    labels_image_t1w.data[np.where(labels_image_t1w.data == val)] = new_val
                    
            # Save segmentations with new labels in output folder
            out_file = labels_file.replace(labels_folder, output_folder)
            if not os.path.exists(os.path.dirname(out_file)):
                os.makedirs(os.path.dirname(out_file))
            labels_image.save(out_file)
            shutil.copy(labels_file.replace('.nii.gz', '.json'), out_file.replace('.nii.gz', '.json'))
            update_json_file(out_file.replace('.nii.gz', '.json'))

            with open(f'{output_folder}/check.txt', 'a') as f:
                name = sub_acq if not 'lowresSag' in sub_acq else f'{sub_acq} '
                f.write(f"{name} : " + " ".join([new_val_list[i] for i in np.argsort(new_val_pos)]) + "\n")
            
            if new_val_list != tss_list:
                with open(f'{output_folder}/new.txt', 'a') as f:
                    name = sub_acq if not 'lowresSag' in sub_acq else f'{sub_acq} '
                    f.write(f"{name}\n")

            if not labels_image_t1w is None:
                out_file_t1w = labels_file_t1w.replace(labels_folder, output_folder)
                if not os.path.exists(os.path.dirname(out_file_t1w)):
                    os.makedirs(os.path.dirname(out_file_t1w))
                labels_image_t1w.save(out_file_t1w)
                shutil.copy(labels_file_t1w.replace('.nii.gz', '.json'), out_file_t1w.replace('.nii.gz', '.json'))
                update_json_file(out_file_t1w.replace('.nii.gz', '.json'))
        

                

mapping_spineps = {
    "T7":14,
    "T8":15,
    "T9":16,
    "T10":17,
    "T11":18,
    "T12":19,
    "T13":28,
    "L1":20,
    "L2":21,
    "L3":22,
    "L4":23,
    "L5":24,
    "L6":25,
    "sacrum":26,
    "T7-T8":114,
    "T8-T9":115,
    "T9-T10":116,
    "T10-T11":117,
    "T11-T12":118,
    "T12-L1":119,
    "L1-L2":120,
    "L2-L3":121,
    "L3-L4":122,
    "L4-L5":123,
    "L5-S":124,
    "L6-S":125,
    "error":126
}

rev_spineps_mapping = {v:k for k,v in mapping_spineps.items()}

new_mapping = {
    "C2":12,
    "C3":13,
    "C4":14,
    "C5":15,
    "C6":16,
    "C7":17,
    "T1":21,
    "T2":22,
    "T3":23,
    "T4":24,
    "T5":25,
    "T6":26,
    "T7":27,
    "T8":28,
    "T9":29,
    "T10":30,
    "T11":31,
    "T12":32,
    "L1":41,
    "L2":42,
    "L3":43,
    "L4":44,
    "L5":45,
    "L6":46,
    "sacrum":50,
    "C2-C3":63,
    "C3-C4":64,
    "C4-C5":65,
    "C5-C6":66,
    "C6-C7":67,
    "C7-T1":71,
    "T1-T2":72,
    "T2-T3":73,
    "T3-T4":74,
    "T4-T5":75,
    "T5-T6":76,
    "T6-T7":77,
    "T7-T8":78,
    "T8-T9":79,
    "T9-T10":80,
    "T10-T11":81,
    "T11-T12":82,
    "T12-L1":91,
    "L1-L2":92,
    "L2-L3":93,
    "L3-L4":94,
    "L4-L5":95,
    "L5-L6":96,
    "L5-S":100
}

rev_new_mapping = {v:k for k,v in new_mapping.items()}

old_mapping = {
    "C2":40,
    "C3":39,
    "C4":38,
    "C5":37,
    "C6":36,
    "C7":35,
    "T1":34,
    "T2":33,
    "T3":32,
    "T4":31,
    "T5":30,
    "T6":29,
    "T7":28,
    "T8":27,
    "T9":26,
    "T10":25,
    "T11":24,
    "T12":23,
    "L1":22,
    "L2":21,
    "L3":20,
    "L4":19,
    "L5":18,
    "sacrum":92,
    "C2-C3":224,
    "C3-C4":223,
    "C4-C5":222,
    "C5-C6":221,
    "C6-C7":220,
    "C7-T1":219,
    "T1-T2":218,
    "T2-T3":217,
    "T3-T4":216,
    "T4-T5":215,
    "T5-T6":214,
    "T6-T7":213,
    "T7-T8":212,
    "T8-T9":211,
    "T9-T10":210,
    "T10-T11":209,
    "T11-T12":208,
    "T12-L1":207,
    "L1-L2":206,
    "L2-L3":205,
    "L3-L4":204,
    "L4-L5":203,
    "L5-S":202
}

rev_old_mapping = {v:k for k,v in old_mapping.items()}

if __name__ == '__main__':
    main()