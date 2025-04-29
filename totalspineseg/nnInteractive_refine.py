from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

# Define constants
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = str(Path(r"SlicerNNInteractive\server\.nninteractive_weights"))

# Define paths
base_path = Path("nnInteractive_test")
img_path = base_path / "totalspineseg" / "input" / "testimg_0000.nii.gz"
seg_path = base_path / "totalspineseg" / "step1_output" / "testimg.nii.gz"
output_path = base_path / "testimg_refine.nii.gz"


def refine_segmentation_single(session, seg_data, seg_label, seg_label_neg, iterations, lasso=True):

    if seg_label not in seg_data:
        return None

    # Extract labels from segmentation mask
    scribble = (seg_data == seg_label).astype(np.uint8)

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
    img_nib = nib.load(str(img_path))
    img_data = img_nib.get_fdata()
    img_data = np.expand_dims(img_data, axis=0)  # Add channel dimension (1, x, y, z)

    # Load segmentation mask
    print(f"Loading segmentation mask from {seg_path}")
    seg_nib = nib.load(str(seg_path))
    seg_data = np.asanyarray(seg_nib.dataobj).round().astype(np.uint8)

    #####################################
    # Set image to the session
    #####################################

    # Set image to the session
    print("Setting image to session...")
    session.set_image(img_data)

    results = (seg_data == 1).astype(np.uint8)  # Initialize results with the spincal cord label

    # for spinal canal segmentation spinal cord is negative label with 1 iteration and usin scribbler not lasso
    print("Refining spinal canal segmentation...")
    cur_redults = refine_segmentation_single(session, seg_data, 2, 1, 1)
    if cur_redults is not None:
        results[cur_redults == 1] = 2

    # for C1 segmentation - !!!not working well!!!
    print("Refining C1 segmentation...")
    cur_redults = refine_segmentation_single(session, seg_data, 11, 0, 1)
    if cur_redults is not None:
        results[cur_redults == 1] = 11

    # for each vertebral label
    print("Refining vertebral segmentation...")
    for s in [12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 41, 42, 43, 44, 45, 50]:
        print(f"Refining segmentation for label {s}...")
        # Set target buffer to zero
        cur_redults = refine_segmentation_single(session, seg_data, s, 0, 5)
        if cur_redults is not None:
            results[cur_redults == 1] = s

    # for each IVD label
    print("Refining IVD segmentation...")
    for s in [63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95, 100]:
        print(f"Refining segmentation for label {s}...")
        # Set target buffer to zero
        cur_redults = refine_segmentation_single(session, seg_data, s, 0, 7)
        if cur_redults is not None:
            results[cur_redults == 1] = s

    ##########################################
    # Save results
    ##########################################

    # Save results
    print(f"Saving results to {output_path}")
    result_nib = nib.Nifti1Image(results, img_nib.affine, img_nib.header)
    result_nib.set_data_dtype(np.uint8)
    result_nib.to_filename(str(output_path))

    print("Segmentation refinement complete!")

################################################
# Download model
################################################

if not Path(DOWNLOAD_DIR).exists():
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
    torch_n_threads=8,  # Adjust based on your CPU
    do_autozoom=True,
    use_pinned_memory=True,
)

# Load the model
model_path = Path(DOWNLOAD_DIR) / MODEL_NAME
print(f"Loading model from {model_path}")
session.initialize_from_trained_model_folder(str(model_path))

refine_segmentation(session, img_path, seg_path, output_path)