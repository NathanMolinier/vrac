#!/bin/bash

# Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# GET PARAMS
# ======================================================================================================================
# SET DEFAULT VALUES FOR PARAMETERS.
# ----------------------------------------------------------------------------------------------------------------------
BIDS_FOLDER="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/whole-spine"
IMG_FOLDER="derivatives/img"
PRED_FOLDER="derivatives/pred"
LABEL_FOLDER="derivatives/labels_nnInteractive"
QC_CANAL_FOLDER="derivatives/qc_canal"
QC_SPINE_FOLDER="derivatives/qc_spine"

# ======================================================================================================================
# SCRIPT STARTS HERE
# ======================================================================================================================

cd "$BIDS_FOLDER"

# Create folders
mkdir -p "$IMG_FOLDER"
mkdir -p "$PRED_FOLDER"
mkdir -p "$LABEL_FOLDER"

# Copy images in same folder
cp $(find sub-amuAL -type f -name '*.nii.gz' | grep -v .git) "$IMG_FOLDER"

# Run totalspineseg on BIDS dataset
source /usr/local/miniforge3/etc/profile.d/conda.sh
conda activate tss_env
echo Running TotalSpineSeg
echo 
# totalspineseg "$IMG_FOLDER" "$PRED_FOLDER" -k step2_output
conda deactivate

# Run nnInteractive using the predictions
for file in $(ls "$IMG_FOLDER");do
    sub=$(echo "$file" | cut -d _ -f 1)
    file_noext=$(echo "$file" | cut -d . -f 1)
    # Create directory
    mkdir -p "$LABEL_FOLDER"/"$sub"/anat

    # Run nnInteractive
    conda activate nnInteractive
    # python ~/data_nvme_p118739/code/vrac/totalspineseg/nnInteractive_refine.py -i "$IMG_FOLDER"/"$file" -s "$PRED_FOLDER"/step2_output/"$file" -o "$LABEL_FOLDER"/"$sub"/anat

    # Create JSON sidecars
    CANAL_NEW="$LABEL_FOLDER"/"$sub"/anat/"$file_noext"_label-canal_seg.json
    python ~/data_nvme_p118739/code/vrac/totalspineseg/create_jsonsidecars.py -path-json "$CANAL_NEW"

    SPINE_NEW="$LABEL_FOLDER"/"$sub"/anat/"$file_noext"_label-spine_dseg.json
    python ~/data_nvme_p118739/code/vrac/totalspineseg/create_jsonsidecars.py -path-json "$SPINE_NEW"
    conda deactivate

    # Replace canal if Abel already corrected
    CANAL_OLD=derivatives/labels/"$sub"/anat/"$file_noext"_label-canal_seg
    if $(cat "$CANAL_OLD".json | grep -q "Abel Salmona"); 
    then 
        cp "$CANAL_OLD"* "$LABEL_FOLDER"/"$sub"/anat;
    fi

    # QC canal
    sct_qc -i "$IMG_FOLDER"/"$file" -s "$LABEL_FOLDER"/"$sub"/anat/"$file_noext"_label-canal_seg.nii.gz -p sct_deepseg_sc -qc "$QC_CANAL_FOLDER"

    # QC spine
    conda activate /home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/spinalcordtoolbox/python/envs/venv_sct 
    python ~/data_nvme_p118739/code/vrac/totalspineseg/qc_tss.py -i "$IMG_FOLDER"/"$file" -s "$LABEL_FOLDER"/"$sub"/anat/"$file_noext"_label-spine_dseg.nii.gz -o "$QC_SPINE_FOLDER"
    conda deactivate

done
    
