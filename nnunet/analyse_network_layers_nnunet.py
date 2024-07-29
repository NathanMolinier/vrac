from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from custom_predictor_nnunet import nnUNetPredictor
import os

def main():
    nnunet_model1 = 'Dataset348_DiscsVertebrae/nnUNetTrainerFT__new_plans__3d_fullres'
    nnunet_model2 = 'Dataset146_nako_manual_inference_plus_spider_143/nnUNetTrainer__nnUNetPlans__3d_fullres'
    test_image = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/test-totalSpineSeg/raw/sub-242186_acq-sagittal_T2w.nii.gz' # SPINEPS good
    #test_image = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/test-totalSpineSeg/raw/sub-spineGeneric003_T1w.nii.gz' # totalSpineSeg good
    out_folder = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/test-totalSpineSeg/test/'
    out_image = os.path.join(out_folder, os.path.basename(test_image).replace('.nii.gz', '_label-spine_dseg.nii.gz'))

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
    predictor.initialize_from_combined_weights(
        join(nnUNet_results, nnunet_model1),
        join(nnUNet_results, nnunet_model2),
        fold1=0,
        fold2=0,
        checkpoint1_name='checkpoint_best.pth',
        checkpoint2_name='checkpoint_final.pth',
        alpha=0.8
    )

    # use list of files as inputs
    predictor.predict_from_files([[test_image]],
                                    [out_image],
                                    save_probabilities=False, overwrite=False,
                                    num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

if __name__ == '__main__':
    main()