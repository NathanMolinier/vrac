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

from transforms.transforms import ConvTransform, HistogramEqualTransform, FunctionTransform, ImageFromSegTransform, RedistributeTransform, ArtifactTransform, SpatialCustomTransform, ShapeTransform

class nnUNetTrainerDAExt(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 10
    
    @staticmethod
    def get_training_transforms(
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
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        ### Adds custom nnunet transforms
        ## Contrast transforms
        # Scharr filter
        transforms.append(RandomTransform(
            ConvTransform(
                kernel_type='Scharr', 
                absolute=True, 
                retain_stats=retain_stats
            ), apply_probability=0.15
        ))
        
        # Apply functions
        func_list = [
            lambda x:torch.log(1 + x), # Log
            torch.sqrt, # sqrt
            torch.sin, # sin
            torch.exp, # exp
            lambda x:1/(1 + torch.exp(-x)), # sig
        ]

        for func in func_list:
            transforms.append(RandomTransform(
                FunctionTransform(
                    function=func,
                    retain_stats=retain_stats
                ), apply_probability=0.05
            ))

        # Histogram equalization
        transforms.append(RandomTransform(
            HistogramEqualTransform(
                retain_stats=retain_stats
            ), apply_probability=0.1
        ))

        # Image from segmentation
        transforms.append(RandomTransform(
            ImageFromSegTransform(
                retain_stats=retain_stats
            ), apply_probability=0
        ))
        
        # Redistribute segmentation values
        transforms.append(RandomTransform(
            RedistributeTransform(
                retain_stats=retain_stats
            ), apply_probability=0.5
        ))

        ## Shape transforms
        transforms.append(RandomTransform(
            ShapeTransform(
                shape_min=1, 
                ignore_axes=(1,2)
            ), apply_probability=0.4
        ))
        
        ## Artifacts generation
        # Motion, Ghosting, Spike, Bias field, Blur, Noise, Swap
        transforms.append(RandomTransform(
            ArtifactTransform(
                motion=True,
                ghosting=True,
                spike=True,
                bias_field=True,
                blur=True,
                noise=True,
                swap=False,
                random_pick=True
            ), apply_probability=0.7
        ))

        ## Spatial transforms
        # Flip, Affine, Elastic, Anisotropy
        transforms.append(RandomTransform(
            SpatialCustomTransform(
                flip=True,
                affine=True,
                elastic=True,
                anisotropy=True,
                random_pick=True
            ), apply_probability=0.6
        ))
        ### End of customs

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.3, 1),
                synchronize_channels=True,
                synchronize_axes=False,
                ignore_axes=(),
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.20
        )) ## Updated loss
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


if __name__=='__main__':
    import cv2
    from vrac.data_management.image import Image
    from vrac.utils.utils import normalize

    plans = load_json('/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/totalspineseg_data/nnUNet/results/r20241115/Dataset101_TotalSpineSeg_step1/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans_small__3d_fullres/plans.json')
    configuration = '3d_fullres'
    fold = 0
    dataset_json = load_json('/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/totalspineseg_data/nnUNet/results/r20241115/Dataset101_TotalSpineSeg_step1/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans_small__3d_fullres/dataset.json')
    device = torch.device('cpu')
    trainer = nnUNetTrainerDAExt(plans, configuration, fold, dataset_json, device)
    
    rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    
    retain_stats = True
    
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
            ignore_label = trainer.label_manager.ignore_label,
            retain_stats = retain_stats
    )

    # Load image
    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/romane_tss_data/nnUNet/raw/Dataset101_TotalSpineSeg_step1/imagesTr/sub-amuFR_T2w_0000.nii.gz'
    img = Image(img_path).change_orientation('RSP')
    img_tensor = torch.from_numpy(img.data.copy()).unsqueeze(0).to(torch.float32)

    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/romane_tss_data/nnUNet/raw/Dataset101_TotalSpineSeg_step1/labelsTr/sub-amuFR_T2w.nii.gz'
    seg = Image(seg_path).change_orientation('RSP')
    seg_tensor = torch.from_numpy(seg.data.copy()).unsqueeze(0)

    tensor_dict = {}
    for i in range(24):
        tensor_dict[f'transfo_{str(i+1)}'] = transforms(**{'image': img_tensor.detach().clone(), 'segmentation': seg_tensor.detach().clone()})
    
    nb_img = len(tensor_dict.keys())
    nb_col = 6
    output = []
    line = []
    aug = [[]]
    for idx, (augment, dic) in enumerate(tensor_dict.items()):
        if len(line) < nb_col:
            img = 255*normalize(tensor_dict[augment]['image'].detach().numpy()[0,85])
            line.append(img)
            aug[-1].append(augment)
        else:
            output.append(np.concatenate(line, axis=1))
            img = 255*normalize(tensor_dict[augment]['image'].detach().numpy()[0,85])
            line = [img]
            aug.append([augment])
    output.append(np.concatenate(line, axis=1))

    out_img = np.concatenate(output, axis=0)
    cv2.imwrite('transforms.png' if retain_stats else 'transforms_nostats.png', out_img)
    print(aug)
