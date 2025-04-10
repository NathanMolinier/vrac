"""
This script is based on https://github.com/ivadomed/utilities/blob/main/dataset_conversion/convert_bids_to_nnUNetV2.py

Converts BIDS-structured dataset to the nnUNetv2 dataset format. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Naga Karthik, Jan Valosek, Théo Mathieu modified by Nathan Molinier
"""
import argparse
import pathlib
from pathlib import Path
import json
import os
from collections import OrderedDict
from loguru import logger
import numpy as np
from progress.bar import Bar

from vrac.data_management.utils import fetch_subject_and_session, fetch_contrast
from vrac.data_management.image import Image


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert cofig file to nnUNetV2 database format.')
    parser.add_argument('--config', required=True, help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--path-out', required=True, help='Path to output directory. Example: ~/data/dataset-nnunet (Required)')
    parser.add_argument('--dataset-name', '-dname', required=True, type=str,
                        help='Specify the task name.')
    parser.add_argument('--dataset-number', '-dnum', required=True, type=int,
                        help='Specify the task number.')
    return parser


def convert_subjects(list_labels, path_out_images, path_out_labels, datasets_path):
    """Function to get image from original BIDS dataset modify if needed and place
        it with a compatible name in nnUNet dataset.

    Args:
        list_labels (list): List containing dictionaries with the paths of images and label in BIDS format.
        path_out_images (str): path to the images directory in the new dataset (test or train).
        path_out_labels (str): path to the labels directory in the new dataset (test or train).

    Returns:
        unique_labels (list[int]): unique values present in segmentations

    """

    # Init progression bar
    bar = Bar('Convert data ', max=len(list_labels))

    # Extract seg values
    unique_labels = []

    for dic in list_labels:
        img_path = os.path.join(datasets_path, dic['IMAGE'])
        label_path = os.path.join(datasets_path, dic['LABEL'])
        dataset_name = dic['IMAGE'].split('/')[0]

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f'Error while loading subject\n {img_path} or {label_path} might not exist --> skipping subject')
        else:
            # Load and reorient image and label to LPI
            label = Image(label_path).change_orientation('LPI')
            img = Image(img_path).change_orientation('LPI')

            if img.data.shape == label.data.shape:

                # Extract information from the img_path
                sub_name, sessionID, filename, modality, echoID, acquisition = fetch_subject_and_session(img_path)

                # Extract contrast
                contrast = fetch_contrast(img_path) 

                # Create new nnunet paths
                nnunet_label_path = os.path.join(path_out_labels, f"{dataset_name}_{sub_name}_{contrast}.nii.gz")
                nnunet_img_path = os.path.join(path_out_images, f"{dataset_name}_{sub_name}_{contrast}_0000.nii.gz")
                
                # Update seg values
                unique_labels += np.unique(label.data).tolist()
                unique_labels = np.unique(unique_labels).tolist()

                # Save images
                label.save(nnunet_label_path)
                img.save(nnunet_img_path)
            else:
                print(f'Error while loading subject\n {img_path} and {label_path} don"t have the same shape --> skipping subject')
        # Plot progress
        bar.suffix  = f'{list_labels.index(dic)+1}/{len(list_labels)}'
        bar.next()
    bar.finish()
    return unique_labels


def main():
    parser = get_parser()
    args = parser.parse_args()
    path_out = Path(os.path.join(os.path.abspath(os.path.expanduser(args.path_out)),
                                 f'Dataset{args.dataset_number:03d}_{args.dataset_name}'))
    
    # Read json file and create a dictionary
    with open(args.config, "r") as file:
        config = json.load(file)
    
    # Check config type
    if config['TYPE'] != "LABEL":
        raise ValueError(f"Config TYPE {config['TYPE']} not supported.")

    # TODO: Allow multi channel training
    channel_dict = {config['CONTRASTS']: 0}

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    train_labels = config['TRAINING'] + config['VALIDATION']
    test_labels = config['TESTING']

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # Convert training and validation subjects to nnunet format
    train_unique = convert_subjects(
        list_labels=train_labels,
        path_out_images=path_out_imagesTr,
        path_out_labels=path_out_labelsTr,
        datasets_path=config['DATASETS_PATH']
    )

    # Convert testing subjects to nnunet format
    test_unique = convert_subjects(
        list_labels=test_labels,
        path_out_images=path_out_imagesTs,
        path_out_labels=path_out_labelsTs,
        datasets_path=config['DATASETS_PATH']
    )

    nb_img_train = len(train_labels)
    nb_img_test = len(test_labels)

    # Check if values present in testing are present in training
    if not all(np.isin(test_unique, train_unique).tolist()):
        raise ValueError(f'Some values present in testing are not present in training\ntraining : {train_unique}\ntesting : {test_unique}\n')

    logger.info(f"Number of training and validation subjects: {nb_img_train}")
    logger.info(f"Number of test subjects: {nb_img_test}")

    # c.f. dataset json generation
    # In nnUNet V2, dataset.json file has become much shorter. The description of the fields and changes
    # can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson
    # this file can be automatically generated using the following code here:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()

    # The following keys are the most important ones.
    """
    channel_names:
        Channel names must map the index to the name of the channel. For BIDS, this refers to the contrast suffix.
        TODO: allow for more channels
        {
            "0": "FLAIR",
            "1": "T1w",
            "2": "T2",
            "3": "T2w"
        }
    Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based 
        training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training: 
        https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    """

    json_dict['channel_names'] = {v: k for k, v in channel_dict.items()}

    json_dict['labels'] = {"background": 0}
    
    # Adding individual classes
    for i, cl in enumerate(np.sort(train_unique)): # -1 to remove background
        if cl != 0:
            json_dict['labels'][f'class_{i}'] = int(cl)

    json_dict["numTraining"] = nb_img_train

    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"
    json_dict["overwrite_image_reader_writer"] = "SimpleITKIO"

    # create dataset.json
    json.dump(json_dict, open(os.path.join(path_out, "dataset.json"), "w"), indent=4)

if __name__ == '__main__':
    main()