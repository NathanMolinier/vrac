"""
Refine segmentations using nnInteractive. Script created by Yehuda Warszawer and edited by Nathan Molinier
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import argparse, os

from vrac.data_management.image import Image, zeros_like
from skimage.morphology import erosion, dilation, disk, ball, square, cube

def refine_segmentation_single(session, seg_data, seg_label, seg_label_neg, iterations, lasso=True):

    # Extract labels from segmentation mask
    scribble = np.isin(seg_data, seg_label).astype(np.uint8)

    if not np.any(scribble):
        return None
    
    # Dilate
    size = 2
    footprint = ball(size)
    scribble = dilation(scribble, footprint)

    # Add scribble interaction
    print("Adding scribble interaction...")
    for i in range(iterations):
        session.set_target_buffer(torch.zeros(seg_data.shape, dtype=torch.uint8))
        session.reset_interactions()
        if lasso:
            # Add positive lasso interaction
            print(f"Adding positive lasso interaction...")
            session.add_lasso_interaction(scribble, include_interaction=True)
        else:
            # Add positive scribble interaction
            print(f"Adding positive scribble interaction...")
            session.add_scribble_interaction(scribble, include_interaction=True)

        if seg_label_neg > 0:
            # Add negative scribble interaction
            print("Adding negative scribble interaction...")
            session.add_scribble_interaction((seg_data == seg_label_neg).astype(np.uint8), include_interaction=False)

        results = session.target_buffer.clone().numpy().astype(np.uint8)

        # If negative labels are present, set them to zero in the results
        if seg_label_neg > 0:
            results[seg_data == seg_label_neg] = 0
        scribble = results

    return results


def refine_segmentation(session, img_path, seg_path, output_path):
    ###############################################
    # Load input data
    ###############################################

    # Load input image
    print(f"Loading input image from {img_path}")
    img_nib = Image(img_path).change_orientation('RPI')
    img_data = img_nib.data
    img_data = np.expand_dims(img_data, axis=0)  # Add channel dimension (1, x, y, z)

    # Load segmentation mask
    print(f"Loading segmentation mask from {seg_path}")
    seg_nib = Image(seg_path).change_orientation('RPI')
    seg_data = seg_nib.data.astype(np.uint8)

    # Init output files
    out_img = zeros_like(seg_nib)

    #####################################
    # Set image to the session
    #####################################

    # Set image to the session
    print("Setting image to session...")
    session.set_image(img_data)

    # for all unique labels in the segmentation
    labels = [int(x) for x in np.unique(seg_data) if x != 0]
    for l in labels:
        print(f"Refining segmentation for label {l}...")
        # Set target buffer to zero
        results = refine_segmentation_single(session, seg_data, l, 0, 2)
        if results is not None:
            out_img.data[results == 1] = l

    ##########################################
    # Save results
    ##########################################

    # Save results
    out_path = os.path.join(output_path, os.path.basename(img_path).replace('.nii.gz', '_label-refined_dseg.nii.gz'))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"Saving refined segmentation to {out_path}")
    out_img.data = out_img.data.astype(np.uint8)
    out_img.save(out_path)

    print("Segmentation refinement complete!")


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Refine segmentation using nnInteractive.')
    parser.add_argument('--image', '-i', required=True, help='Path to the input image (Required)')
    parser.add_argument('--seg', '-s', required=True, help='Path to the input segmentation (Required)')
    parser.add_argument('--out', '-o', required=True, help='Path of the output folder (Required)')
    return parser


def main():
    # Load parser
    parser = get_parser()
    args = parser.parse_args()

    # Define paths
    img_path = args.image
    seg_path = args.seg
    output_path = args.out

    # Define constants
    REPO_ID = "nnInteractive/nnInteractive"
    MODEL_NAME = "nnInteractive_v1.0"
    DOWNLOAD_DIR = "/home/ge.polymtl.ca/p118739/data/nnInteractive/weights/"

    ################################################
    # Download model
    ################################################

    if not Path(os.path.join(DOWNLOAD_DIR, MODEL_NAME)).exists():
        print("Downloading model...")
        Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
        download_path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=DOWNLOAD_DIR
        )
        print(f"Model downloaded to {download_path}")

    ###########################################
    # Initialize inference session
    ###########################################

    print("Initializing inference session...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=True,
        torch_n_threads=os.cpu_count(),  # Adjust based on your CPU
        do_autozoom=True,
    )

    # Load the model
    model_path = Path(DOWNLOAD_DIR) / MODEL_NAME
    print(f"Loading model from {model_path}")
    session.initialize_from_trained_model_folder(str(model_path))

    refine_segmentation(session, img_path, seg_path, output_path)

if __name__=='__main__':
    main()