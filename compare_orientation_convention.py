from BIDSIFICATION.image import Image
from TPTBox.core.nii_wrapper import NII
from monai.transforms import Compose, LoadImaged, Orientationd
import nibabel as nib

img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-orientation/sub-242236_acq-sagittal_T2w.nii.gz'

# Load image using SCT
sct_image = Image(img_path)
sct_ori = sct_image.orientation
print(f"SCT's orientation is {sct_ori}")

# Load image using the TPTbox
tpt_image = NII.load(img_path, seg=False)
tpt_ori = tpt_image.orientation
print(f"TPT's orientation is {''.join(tpt_ori)}")

# Load image using MONAI transform
monai_dict = {"image":img_path}
transforms = Compose([LoadImaged(keys=["image"])])
monai_image = transforms(monai_dict)['image']

# TPT nibabel use
nib_img = nib.load(img_path)
ort = nib.orientations.io_orientation(nib_img.affine)
nib_ori = nib.orientations.ornt2axcodes(ort)

print(1)


