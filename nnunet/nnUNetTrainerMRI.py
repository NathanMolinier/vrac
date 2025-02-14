from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0_NoMirroring
from typing import Tuple, Union, List
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgenerators.utilities.file_and_folder_operations import load_json

import torch

from vrac.data_management.image import Image

class nnUNetTrainerMRI(nnUNetTrainer):
    
    @staticmethod
    def get_mri_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        
        return ComposeTransforms(transforms)

def mri_transforms():
    
    return ComposeTransforms(transforms)

if __name__=='__main__':
    plans = load_json('/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_preprocessed/Dataset348_DiscsVertebrae/nnUNetPlans.json')
    configuration = '3d_fullres'
    fold = 0
    dataset_json = load_json('/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_preprocessed/Dataset348_DiscsVertebrae/dataset.json')
    device = torch.device('cpu')
    unpack_dataset = True
    trainer = nnUNetTrainerMRI(plans, configuration, fold, dataset_json, unpack_dataset, device)
    
    rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    
    transforms = trainer.get_training_transforms(
            patch_size = trainer.configuration_manager.patch_size,
            rotation_for_DA = rotation_for_DA, 
            deep_supervision_scales = trainer._get_deep_supervision_scales(), 
            mirror_axes = mirror_axes, 
            do_dummy_2d_data_aug = do_dummy_2d_data_aug,
            use_mask_for_norm = trainer.configuration_manager.use_mask_for_norm,
            is_cascaded = trainer.is_cascaded, 
            foreground_labels = trainer.label_manager.foreground_labels,
            regions = trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else None,
            ignore_label = trainer.label_manager.ignore_label
    )

    #mri_transforms()

    # Load image
    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_raw/Dataset348_DiscsVertebrae/imagesTr/sub-WHOLEamuFR_T1w_0000.nii.gz'
    img = Image(img_path).change_orientation('LPI')
    img_tensor = torch.from_numpy(img.data).unsqueeze(0)

    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data/nnUNet_raw/Dataset348_DiscsVertebrae/labelsTr/sub-WHOLEamuFR_T1w.nii.gz'
    seg = Image(seg_path).change_orientation('LPI')
    seg_tensor = torch.from_numpy(seg.data).unsqueeze(0)

    #tmp = mri_transforms(**{'image': img_tensor, 'segmentation': seg_tensor})
    tmp = transforms(**{'image': img_tensor, 'segmentation': seg_tensor})
    print()
