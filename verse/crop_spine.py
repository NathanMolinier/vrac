from vrac.data_management.image import Image
import os, json, argparse, shutil
from progress.bar import Bar
import numpy as np
import cv2

from vrac.utils.utils import normalize


VERT_DICT = {
    "C1":1,
    "C2":2,
    "C3":3,
    "C4":4,
    "C5":5,
    "C6":6,
    "C7":7,
    "T1":8,
    "T2":9,
    "T3":10,
    "T4":11,
    "T5":12,
    "T6":13,
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
    "sacrum":[26, 29, 30, 31, 32],
    "cocygis":27
}

def main():
    # Load variables
    config_raw_path = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/verse-1mm-round/verse-spine-ct-1mm.json'
    out_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/verse-1mm-round-crop'

    # Load json config
    with open(config_raw_path, "r") as file:
        config = json.load(file)
    
    t13_list = []
    t11_list = []

    # Create output folders
    if not os.path.exists(out_folder):
        os.makedirs(os.path.join(out_folder, 'image'))
        os.makedirs(os.path.join(out_folder, 'label'))
    
    for split in ['TRAINING', 'VALIDATION', 'TESTING']:
        dict_list = config[split]
        bar = Bar(f'Deal with {split} data', max=len(dict_list))
        for dic in dict_list:
            # Init paths
            label_path = os.path.join(config['DATASETS_PATH'], dic['LABEL'])
            img_path = os.path.join(config['DATASETS_PATH'], dic['IMAGE'])

            # Load data
            label = Image(label_path)
            img = Image(img_path)

            # Reorient data to RSP
            label.change_orientation('RSP')
            img.change_orientation('RSP')

            # Check max and min label
            label_order = [v for k,v in VERT_DICT.items() if k not in ["sacrum", "cocygis"]] + [26, 29, 30, 31, 32, 27]
            unique_labels = np.unique(label.data).tolist()
            unique_labels.remove(0)

            max_label = 0
            min_label = 0
            i = 0
            while min_label == 0:
                if label_order[i] in unique_labels:
                    min_label = label_order[i]
                i += 1
            
            j = len(label_order)-1
            while max_label == 0:
                if label_order[j] in unique_labels:
                    max_label = label_order[j]
                j -= 1

            shape = label.data.shape

            print(f"\nUnique list: {unique_labels}")
            print(f"min label: {min_label}")
            print(f"max label: {max_label}\n")

            # Check if T13 or T11 case
            if 28 in unique_labels: # T13
                t13_list.append(dic['IMAGE'])

            if np.isin(unique_labels, [18, 20]).all() and 19 not in unique_labels: # T11
                t11_list.append(dic['IMAGE'])
            
            # Define max and min coordinate for cropping along z axis
            if min_label == 1:
                min_z = 0 # Take the full image
            else:
                # Mask min structure
                min_list = []
                for r in range(shape[0]): # RL axis
                    for p in range(shape[2]): # AP axis
                        z_indices = np.where(label.data[r, :, p] == min_label)[0]
                        if len(z_indices) > 0:
                            # Extract the min for each column to extract a surface
                            min_list.append(np.min(z_indices))
                
                # Extract max
                min_z = np.max(min_list)
            
            if max_label == 27:
                max_z = shape[1] # Take the full image
            else:
                # Mask max structure
                shape = label.data.shape
                max_list = []
                for r in range(shape[0]): # RL axis
                    for p in range(shape[2]): # AP axis
                        z_indices = np.where(label.data[r, :, p] == max_label)[0]
                        if len(z_indices) > 0:
                            # Extract the min for each column to extract a surface
                            max_list.append(np.max(z_indices))
                
                # Extract min
                max_z = np.min(max_list)

            # Crop images to avoid having unsegmented vertebrae
            crop_label = crop(label, xmin=0, xmax=shape[0], ymin=min_z, ymax=max_z, zmin=0, zmax=shape[2])
            crop_img = crop(img, xmin=0, xmax=shape[0], ymin=min_z, ymax=max_z, zmin=0, zmax=shape[2])

            # Save cropped image and label
            crop_label.save(os.path.join(out_folder, 'label', os.path.basename(dic['LABEL'])))
            crop_img.save(os.path.join(out_folder, 'image', os.path.basename(dic['IMAGE'])))

            # Plot progress
            bar.suffix  = f'{dict_list.index(dic)+1}/{len(dict_list)}'
            bar.next()
        bar.finish()

    with open(os.path.join(out_folder, 't13.txt'), "w") as f:
        f.writelines(t13_list)

    with open(os.path.join(out_folder, 't11.txt'), "w") as f:
        f.writelines(t11_list)

    shutil.copy(config_raw_path, out_folder)


def crop(img_in, xmin, xmax, ymin, ymax, zmin, zmax):
    data_crop = img_in.data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    img_out = Image(param=data_crop, hdr=img_in.hdr)
    # adapt the origin in the qform matrix
    new_origin = np.dot(img_out.hdr.get_qform(), [xmin, ymin, zmin, 1])
    img_out.hdr.structarr['qoffset_x'] = new_origin[0]
    img_out.hdr.structarr['qoffset_y'] = new_origin[1]
    img_out.hdr.structarr['qoffset_z'] = new_origin[2]
    # adapt the origin in the sform matrix
    new_origin = np.dot(img_out.hdr.get_sform(), [xmin, ymin, zmin, 1])
    img_out.hdr.structarr['srow_x'][-1] = new_origin[0]
    img_out.hdr.structarr['srow_y'][-1] = new_origin[1]
    img_out.hdr.structarr['srow_z'][-1] = new_origin[2]
    return img_out

if __name__=='__main__':
    main()