from vrac.data_management.image import Image
import os


def main():
    img_4d_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/fast-mri/3D/niftii/08-localizer_fast_FA3_TR10_BW240_5slc_15meas_p2_1.5mm_localizer_fast_FA3_TR10_BW240_5slc_15meas_p2_1.5mm_20260120091318_8.nii.gz"
    out_folder = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/fast-mri/3D/niftii/split_08/"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    img_4d = Image(img_4d_path).change_orientation('RPI')

    data = img_4d.data
    img_name = os.path.basename(img_4d_path).split('.')[0]

    for i in range(data.shape[-1]):
        out_path = os.path.join(out_folder, f"{img_name}_meas{i+1:03}.nii.gz")
        Image(data[:,:,:,i], hdr=img_4d.hdr).save(out_path)

if __name__=='__main__':
    main()