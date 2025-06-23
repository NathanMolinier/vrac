from vrac.data_management.image import Image
import os


def main():
    img_4d_path = "/Users/nathan/data/fast-headPos/_localizer_fast_FA3_TR10_BW240_20meas_20250604135044_28.nii"
    out_folder = "/Users/nathan/data/fast-headPos/split/"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    img_4d = Image(img_4d_path).change_orientation('RPI')

    data = img_4d.data

    for i in range(data.shape[-1]):
        out_path = os.path.join(out_folder, f"localizer_fast_FA3_TR10_BW240_meas{i+1:03}.nii.gz")
        Image(data[:,:,:,i], hdr=img_4d.hdr).save(out_path)

if __name__=='__main__':
    main()