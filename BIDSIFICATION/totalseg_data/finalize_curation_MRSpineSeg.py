import os
import sys
import shutil
import json
import argparse
import logging
import datetime
import csv
import time
import subprocess
import numpy as np

from vrac.data_management.image import Image, zeros_like


# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

def get_parser():
    parser = argparse.ArgumentParser(description='Convert dataset to BIDS format.')
    parser.add_argument("-i", "--path-dataset",
                        help="Path to the BIDS dataset.",
                        required=True)
    parser.add_argument("-name", type=str,
                        help="Dataset name BIDS dataset.",
                        required=True)
    return parser


def create_json_file(path_file_out):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    path_json_out = path_file_out.replace('.nii.gz', '.json')
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "Manual",
                "Description": "Manually labeled for the MRSpineSeg Challenge https://paperswithcode.com/dataset/mrspineseg-challenge"
            }
        ]
    }
    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        logger.info(f'Created: {path_json_out}')


def write_json(path_output, json_filename, data_json):
    """
    :param path_output: path to the output BIDS folder
    :param json_filename: json filename, for example: participants.json
    :param data_json: JSON formatted content
    :return:
    """
    with open(os.path.join(path_output, json_filename), 'w') as json_file:
        json.dump(data_json, json_file, indent=4)
        # Add last newline
        json_file.write("\n")
        logger.info(f'{json_filename} created in {path_output}')
        

def create_participants_tsv(participants_tsv_list, path_output):
    """
    Write participants.tsv file
    :param participants_tsv_list: list containing [subject_out, pathology_out, subject_in, centre_in, centre_out],
    example:[sub-torontoDCM001, DCM, 001, 01, toronto]
    :param path_output: path to the output BIDS folder
    :return:
    """
    with open(os.path.join(path_output, 'participants.tsv'), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['participant_id', 'source_id', 'species', 'age', 'sex', 'pathology', 'institution', 'notes'])

        # Species
        species = ['homo sapiens']

        # Add missing information age --> notes
        missing_data = ['n/a']*5

        # Add rows to tsv file
        participants_tsv_list = sorted(participants_tsv_list, key=lambda a : a[0])
        for item in participants_tsv_list:
            tsv_writer.writerow(list(item) + species + missing_data)
        logger.info(f'participants.tsv created in {path_output}')


def create_participants_json(path_output):
    """
    Create participants.json file
    :param path_output: path to the output BIDS folder
    :return:
    """
    # Create participants.json
    data_json = {
        "participant_id": {
            "Description": "Unique Participant ID",
            "LongName": "Participant ID"
        },
        "source_id": {
            "Description": "Original subject name"
        },
        "species": {
            "Description": "Binomial species name of participant",
            "LongName": "Species"
        },
        "age": {
            "Description": "Participant age",
            "LongName": "Participant age",
            "Units": "years"
        },
        "sex": {
            "Description": "sex of the participant as reported by the participant",
            "Levels": {
                "M": "male",
                "F": "female",
                "O": "other"
            }
        },
        "pathology": {
            "Description": "The diagnosis of pathology of the participant",
            "LongName": "Pathology name",
            "Levels": {
                "HC": "Healthy Control",
                "DCM": "Degenerative Cervical Myelopathy (synonymous with CSM - Cervical Spondylotic Myelopathy)",
                "MildCompression": "Asymptomatic cord compression, without myelopathy",
                "MS": "Multiple Sclerosis",
                "SCI": "Traumatic Spinal Cord Injury"
            }
        },
        "institution": {
            "Description": "Human-friendly institution name",
            "LongName": "BIDS Institution ID"
        },
        "notes": {
            "Description": "Additional notes about the participant. For example, if there is more information about a disease, indicate it here.",
            "LongName": "Additional notes"
        }
    }
    write_json(path_output, 'participants.json', data_json)
    

def create_dataset_description(path_output, dataset_name, derivative=False):
    """
    Create dataset_description.json file
    :param path_output: path to the output BIDS folder
    :return:
    """
    if not derivative:
        data_json = {
            "BIDSVersion": "BIDS 1.9.0",
            "Name": dataset_name,
            "DatasetType": "raw"
        }
    else:
        data_json = {
            "BIDSVersion": "BIDS 1.9.0",
            "Name": dataset_name,
            "DatasetType": "derivative"
        }
    write_json(path_output, 'dataset_description.json', data_json)


def copy_script(path_output):
    """
    Copy the script itself to the path_output/code folder
    :param path_output: path to the output BIDS folder
    :return:
    """
    path_script_in = sys.argv[0]
    path_code = os.path.join(path_output, 'code')
    if not os.path.isdir(path_code):
        os.makedirs(path_code, exist_ok=True)
    path_script_out = os.path.join(path_code, 'curate.py')
    logger.info(f'Copying {path_script_in} to {path_script_out}')
    shutil.copyfile(path_script_in, path_script_out)


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Make sure that input args are absolute paths
    path_dataset = os.path.abspath(args.path_dataset)

    # Dataset name 
    dataset_name = args.name

    # Check if input path is valid
    if not os.path.isdir(path_dataset):
        print(f'ERROR - {path_dataset} does not exist.')
        sys.exit()

    FNAME_LOG = os.path.join(path_dataset, 'bids_conversion.log')
    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    
    fh = logging.FileHandler(os.path.join(os.path.abspath(os.curdir), FNAME_LOG))
    logging.root.addHandler(fh)
    logger.info("INFO: log file will be saved to {}".format(FNAME_LOG))

    # Print current time and date to log file
    logger.info('\nAnalysis started at {}'.format(datetime.datetime.now()))
    
    # Initialize dict for participants.tsv
    if os.path.exists(os.path.join(path_dataset, 'participants.tsv')):
        with open(os.path.join(path_dataset, 'participants.tsv')) as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')
            sub_dict_tsv = dict(reader)
            del sub_dict_tsv['participant_id']
    else:
        sub_dict_tsv = dict()
    
    # Fix subjects numbering
    for root, dirs, files in os.walk(path_dataset, topdown=True):
        for d in dirs:
            file_path = os.path.join(root, d)
            if 'sub' in file_path and os.path.exists(file_path):
                num = int(file_path.split('sub-')[1].split('/')[0])
                old_sub = f'sub-{str(num)}'
                num_str = f'{num:03d}'
                new_sub = f'sub-{num_str}'
                new_file_path = file_path.replace(old_sub, new_sub)
                if file_path != new_file_path:
                    shutil.copytree(file_path, new_file_path)
                    shutil.rmtree(file_path)
    
    for root, dirs, files in os.walk(path_dataset, topdown=False):
        for f in files:
            file_path = os.path.join(root, f)
            if 'sub' in file_path:
                num = int(file_path.split('sub-')[1].split('/')[0])
                old_sub = f'sub-{str(num)}'
                num_str = f'{num:03d}'
                new_sub = f'sub-{num_str}'
                new_file_path = file_path.replace(old_sub, new_sub)
                if file_path != new_file_path:
                    os.rename(file_path, new_file_path)

    # Fix image size, derivatives name and add json files
    derivative_path = 'derivatives/labels/'
    for sub in os.listdir(os.path.join(path_dataset, derivative_path)):
        path_sub_der_dir = os.path.join(path_dataset, derivative_path, sub, 'anat')
        if os.path.isdir(path_sub_der_dir):
            for file_name in os.listdir(path_sub_der_dir):
                if '.nii.gz' in file_name:
                    file_path = os.path.join(path_sub_der_dir, file_name)
                    if '_seg' in file_name:
                        # Register image to fix slight different s-form and q-form
                        img_path = os.path.join(path_dataset, sub, 'anat', file_name.replace('_seg',''))
                        subprocess.check_call([
                            'sct_register_multimodal',
                            '-i', file_path,
                            '-d', img_path,
                            '-o', file_path,
                            '-identity', '1',
                            '-x', 'nn'
                        ])

                        # Load image
                        img = Image(img_path).change_orientation('RPI')

                        # Split sacrum and other labels
                        label = Image(file_path).change_orientation('RPI')
                    
                        # Save files
                        spine_path = file_path.replace('_seg', '_label-spine_dseg')
                        label.save(spine_path)
                        img.save(img_path)
                        create_json_file(spine_path)
                    else:
                        raise ValueError('Other files detected without _seg')
            # Add subject name to participant.tsv 
            if sub not in sub_dict_tsv.keys(): # Add only one time each subject into the participant.csv
                # Aggregate subjects for participants.tsv
                orig_name = sub.split('-')[-1]
                sub_dict_tsv[sub] = orig_name
        else:
            raise ValueError(f"{path_sub_der_dir} is not a directory")

    participants_tsv_list = list(sub_dict_tsv.items())                 

    create_participants_tsv(participants_tsv_list, path_dataset)
    create_participants_json(path_dataset)
    create_dataset_description(path_dataset, dataset_name)
    create_dataset_description(os.path.join(path_dataset,'derivatives/labels'), dataset_name, derivative=True)
    copy_script(path_dataset)

if __name__ == "__main__":
    main()          