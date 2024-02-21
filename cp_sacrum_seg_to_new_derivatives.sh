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
PATH_CONFIG="/home/GRAMES.POLYMTL.CA/p118739/data/config_data/add_sacrum.json"
DERIVATIVE_FOLDER="labels"
NEW_DERIVATIVE_FOLDER="labels-sacrum"
LABEL_SUFFIX="_label-sacrum_seg"

# Print variables to allow easier debug
echo "See variables:"
echo "PATH_CONFIG: ${PATH_CONFIG}"
echo "DERIVATIVE_FOLDER: ${DERIVATIVE_FOLDER}"
echo "NEW_DERIVATIVE_FOLDER: ${NEW_DERIVATIVE_FOLDER}"
echo "LABEL_SUFFIX: ${LABEL_SUFFIX}"
echo

# ======================================================================================================================
# SCRIPT STARTS HERE
# ======================================================================================================================
# Fetch datasets path
DATASETS_PATH=$(jq -r '.DATASETS_PATH' ${PATH_CONFIG})

# Go to folder where data will be copied and processed
cd "$DATASETS_PATH"

# Fetch TESTING files
FILES=$(jq -r '.TESTING[]' ${PATH_CONFIG})

# Loop across the files
for FILE_PATH in $FILES; do
    BIDS_FOLDER=$(echo "$FILE_PATH" | cut -d / -f 1)
    IN_FILE_NAME=$(echo "$FILE_PATH" | awk -F / '{print $NF}' )
    SEG_FILE_NAME=${IN_FILE_NAME/".nii.gz"/"${LABEL_SUFFIX}.nii.gz"}
    IMG_PATH=${FILE_PATH/"${BIDS_FOLDER}/"/}
    SUB_PATH=${IMG_PATH/"/${IN_FILE_NAME}"/}
    BIDS_DERIVATIVES="${BIDS_FOLDER}/derivatives/${DERIVATIVE_FOLDER}"
    NEW_BIDS_DERIVATIVES="${BIDS_FOLDER}/derivatives/${NEW_DERIVATIVE_FOLDER}"
    SEG_PATH="${BIDS_DERIVATIVES}/${SUB_PATH}/${SEG_FILE_NAME}"
    NEW_SEG_FOLDER="${NEW_BIDS_DERIVATIVES}/${SUB_PATH}/"
    NEW_SEG_PATH="${NEW_SEG_FOLDER}/${SEG_FILE_NAME}"
    
    # JSON path
    JSON_PATH=${SEG_PATH/".nii.gz"/".json"}
    NEW_JSON_PATH=${NEW_SEG_PATH/".nii.gz"/".json"}

    # Create DERIVATIVES_FOLDER if missing
    if [[ ! -d ${NEW_SEG_FOLDER} ]]; then
        echo "Creating folders $NEW_SEG_FOLDER"
        mkdir -p "${NEW_SEG_FOLDER}"
    fi

    # Copy segmentation
    cp "${SEG_PATH}" "${NEW_SEG_PATH}"
    cp "${JSON_PATH}" "${NEW_JSON_PATH}"

done

