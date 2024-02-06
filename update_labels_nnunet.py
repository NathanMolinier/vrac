from BIDSIFICATION.image import Image
import os
import numpy as np
from progress.bar import Bar

path_nnunet_folder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/nnunet_data/nnUNet_raw/Dataset100_TotalSegMRI'
path_new_folder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/nnunet_data/nnUNet_raw/Dataset102_discs-1class'
one_class = True

label_folder = 'labelsTr'
img_folder = 'imagesTr'
ext = '.nii.gz'
discs_indx = np.arange(25, 48)

labels_path = [os.path.join(path_nnunet_folder, label_folder, f) for f in os.listdir(os.path.join(path_nnunet_folder,label_folder))]

# Init progression bar
bar = Bar('Convert data ', max=len(labels_path))

for label_path in labels_path:
    filename = label_path.split('/')[-1].split(ext)[0]
    img_path = os.path.join(path_nnunet_folder, img_folder, filename + '_0000' + ext)
    new_label_path = os.path.join(path_new_folder, label_folder, filename + ext)
    new_img_path = os.path.join(path_new_folder, img_folder, filename + '_0000' + ext)

    # Create output directories
    if not os.path.exists(os.path.dirname(new_label_path)):
        os.makedirs(os.path.dirname(new_label_path))

    if  not os.path.exists(os.path.dirname(new_img_path)):
        os.makedirs(os.path.dirname(new_img_path))
    
    if os.path.exists(img_path) and os.path.exists(label_path):
        img = Image(img_path)
        label = Image(label_path)
        data = label.data
        new_data = np.zeros_like(data)
        for i, num in enumerate(discs_indx):
            new_data[np.where(data==num)] = int(i + 1) if not one_class else 1
        label.data = new_data
        label.save(new_label_path)
        img.save(new_img_path)
    else:
        raise ValueError(f'{img_path} or {label_path} do not exist')
    
    # Plot progress
    bar.suffix  = f'{labels_path.index(label_path)+1}/{len(labels_path)}'
    bar.next()
bar.finish()
        
    


    
    
    

