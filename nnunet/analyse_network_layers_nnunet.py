from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from custom_predictor_nnunet import nnUNetPredictor

def main():
    nnunet_model = 'Dataset146_nako_manual_inference_plus_spider_143/nnUNetTrainer__nnUNetPlans__3d_fullres'
    test_image = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/test-totalSpineSeg/raw/sub-242186_acq-sagittal_T2w.nii.gz'
    out_image = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/test-totalSpineSeg/test/sub-242186_acq-sagittal_T2w_label-spine_dseg.nii.gz'

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, nnunet_model),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    # use list of files as inputs
    predictor.predict_from_files([[test_image]],
                                    [out_image],
                                    save_probabilities=False, overwrite=False,
                                    num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

if __name__ == '__main__':
    main()