from vrac.data_management.image import Image, zeros_like

def main():
    rootlets_2 = '/Users/nathan/data/lbp-lumbar-usf-2024/derivatives/labels/sub-018/ses-A/anat/sub-018_ses-A_acq-isotropic_T2w_label-rootlets_seg.nii.gz'
    #rootlets_1 = '/Users/nathan/Desktop/test-nninteractive/sub-001_ses-A_acq-isotropic_T2w_label-rootlets_dseg.nii.gz'
    out_rootlets = '/Users/nathan/data/lbp-lumbar-usf-2024/derivatives/labels/sub-018/ses-A/anat/sub-018_ses-A_acq-isotropic_T2w_label-rootlets_seg.nii.gz'

    image_2 = Image(rootlets_2).change_orientation('RPI')
    #image_1 = Image(rootlets_1).change_orientation('RPI')

    out_image = zeros_like(image_2)
    #out_image.data[image_1.data == 1] = 1
    out_image.data[image_2.data > 0] = 1

    out_image.save(out_rootlets)
    
if __name__ == "__main__":
    main()