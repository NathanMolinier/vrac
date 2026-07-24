import os
import sys
import shutil
import json
import argparse
import logging
import datetime
import csv
import re
from vrac.data_management.image import Image

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

CLASSES = {
    "0": "background", 
    "1": "spleen", 
    "2": "right kidney", 
    "3": "left kidney", 
    "4": "gall bladder", 
    "5": "esophagus", 
    "6": "liver", 
    "7": "stomach", 
    "8": "arota", 
    "9": "postcava", 
    "10": "pancreas", 
    "11": "right adrenal gland", 
    "12": "left adrenal gland", 
    "13": "duodenum", 
    "14": "bladder", 
    "15": "prostate/uterus"
}

def get_parser():
    parser = argparse.ArgumentParser(description='Convert lbp-lumbar-stanford dataset to BIDS format.')
    parser.add_argument("-i", "--path-dataset",
                        help="Path to the non-BIDS dataset root (e.g., SpineMRI_Pfirrmann_AG25)",
                        required=True)
    parser.add_argument("-o", "--path-output",
                        help="Path to the output folder where the BIDS dataset will be stored.",
                        required=True)
    return parser


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f'Creating directory: {path}')


def write_json(path_output, json_filename, data_json):
    with open(os.path.join(path_output, json_filename), 'w') as json_file:
        json.dump(data_json, json_file, indent=4)
        json_file.write("\n")
        logger.info(f'{json_filename} created in {path_output}')


def create_participants_tsv(participants, path_output):
    with open(os.path.join(path_output, 'participants.tsv'), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['participant_id', 'source_id', 'species', 'age', 'sex', 'pathology', 'institution', 'notes'])
        species = ['homo sapiens']
        extra_data = ['n/a', 'n/a', 'n/a', 'n/a', 'n/a']
        for item in sorted(participants, key=lambda a: a[0]):
            tsv_writer.writerow(list(item) + species + extra_data)
        logger.info(f'participants.tsv created in {path_output}')


def create_participants_json(path_output):
    data_json = {
        "participant_id": {"Description": "Unique Participant ID", "LongName": "Participant ID"},
        "source_id": {"Description": "Original subject folder name"},
        "species": {"Description": "Binomial species name of participant", "LongName": "Species"},
        "age": {"Description": "Participant age", "LongName": "Participant age", "Units": "years"},
        "sex": {"Description": "sex of the participant as reported by the participant", "Levels": {"M": "male", "F": "female", "O": "other"}},
        "pathology": {"Description": "The diagnosis of pathology of the participant", "LongName": "Pathology name", "Levels": {"LBP": "Low Back Pain"}},
        "institution": {"Description": "Human-friendly institution name", "LongName": "BIDS Institution ID"},
        "notes": {"Description": "Additional notes about the participant", "LongName": "Additional notes"}
    }
    write_json(path_output, 'participants.json', data_json)


def create_dataset_description(path_output):
    data_json = {
        "BIDSVersion": "1.9.0",
        "Name": "abdominal-amos22",
        "DatasetType": "raw"
    }
    create_folder(path_output)
    write_json(path_output, 'dataset_description.json', data_json)

def create_json_sidecar(path_output):
    if path_output.endswith('.nii.gz'):
        path_json = path_output.replace('.nii.gz', '.json')
    else:
        path_json = path_output
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "Semi-automatic",
                "Author": "https://arxiv.org/pdf/2206.08023",
            }
        ],
        "Labels": {
            "0": "background",
            "1": "spleen",
            "2": "right kidney",
            "3": "left kidney",
            "4": "gall bladder",
            "5": "esophagus",
            "6": "liver",
            "7": "stomach",
            "8": "arota",
            "9": "postcava",
            "10": "pancreas",
            "11": "right adrenal gland",
            "12": "left adrenal gland",
            "13": "duodenum",
            "14": "bladder",
            "15": "prostate/uterus"
        }
    }
    write_json(os.path.dirname(path_json), os.path.basename(path_json), data_json)


def copy_script(path_output):
    path_script_in = sys.argv[0]
    path_code = os.path.join(path_output, 'code')
    create_folder(path_code)
    path_script_out = os.path.join(path_code, os.path.basename(path_script_in))
    logger.info(f'Copying {path_script_in} to {path_script_out}')
    shutil.copyfile(path_script_in, path_script_out)


def subject_to_bids(img_path):
    """Extract number from filename"""
    return f"sub-{img_path.replace('_', '').replace('.nii.gz', '')}"


def normalize_filename(subject_bids, label=False):
    """
    Normalize filename by identifying the modality
    """
    val = int(subject_bids.split('amos')[-1])  # Get the part after 'amos'
    if val < 500:
        contrast = 'CT'
    else:
        contrast = 'MR'
    if label:
        out_fname = f"{subject_bids}_{contrast}_label-abdominal_dlabel.nii.gz"
    else:
        out_fname = f"{subject_bids}_{contrast}.nii.gz"
    return out_fname


def main(path_dataset, path_output):
    # Validate inputs
    if not os.path.isdir(path_dataset):
        print(f'ERROR - {path_dataset} does not exist.')
        sys.exit(1)

    create_folder(path_output)

    # Setup logging
    FNAME_LOG = os.path.join(path_output, 'bids_conversion.log')
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(FNAME_LOG)
    logging.root.addHandler(fh)
    logger.info(f"INFO: log file will be saved to {FNAME_LOG}")
    logger.info(f'\nAnalysis started at {datetime.datetime.now()}')

    participants = []
    processed_images = 0
    processed_jsons = 0

    img_dirs = [di for di in os.listdir(path_dataset) if di.startswith('images')]
    label_dirs = [di for di in os.listdir(path_dataset) if di.startswith('labels')]

    # Each immediate child under dataset root is a subject folder (e.g., '51', '56', '81')
    for di in img_dirs:
        for filename in sorted(os.listdir(os.path.join(path_dataset, di))):
            path_file_in = os.path.join(path_dataset, di, filename)
            if not path_file_in.endswith('.nii.gz') or not filename.startswith('amos'):
                print(f"Skipping IMAGE file: {path_file_in}")
                continue

            subject_bids = subject_to_bids(filename)

            anat_dir = os.path.join(path_output, subject_bids, 'anat')
            create_folder(anat_dir)

            # Track participant
            if subject_bids not in [p[0] for p in participants]:
                participants.append((subject_bids, filename.replace('.nii.gz', '')))
            
            normalized_fname = normalize_filename(subject_bids)
            path_file_out = os.path.join(anat_dir, normalized_fname)

            # Image: load, RPI, save
            logger.info(f'Processing image: {path_file_in}')
            img = Image(path_file_in).change_orientation('RPI')
            img.save(path_file_out)
            processed_images += 1
            logger.info(f'Saved: {path_file_out}')

    # Create derivatives for labels
    create_folder(os.path.join(path_output, 'derivatives'))
    for di in label_dirs:
        for filename in sorted(os.listdir(os.path.join(path_dataset, di))):
            path_file_in = os.path.join(path_dataset, di, filename)
            if not path_file_in.endswith('.nii.gz') or not filename.startswith('amos'):
                print(f"Skipping LABEL file: {path_file_in}")
                continue

            subject_bids = subject_to_bids(filename)

            anat_dir = os.path.join(path_output, 'derivatives', subject_bids, 'anat')
            create_folder(anat_dir)

            # Track participant
            if subject_bids not in [p[0] for p in participants]:
                participants.append((subject_bids, filename.replace('.nii.gz', '')))
            
            normalized_fname = normalize_filename(subject_bids, label=True)
            path_file_out = os.path.join(anat_dir, normalized_fname)

            # Image: load, RPI, save
            logger.info(f'Processing label: {path_file_in}')
            img = Image(path_file_in).change_orientation('RPI')
            img.save(path_file_out)
            processed_images += 1
            logger.info(f'Saved: {path_file_out}')

            # Save JSON sidecar for labels
            create_json_sidecar(path_file_out)
            processed_jsons += 1
            logger.info(f'Saved: {path_file_out}')


    # Write metadata
    create_participants_tsv(participants, path_output)
    create_participants_json(path_output)
    create_dataset_description(path_output)
    copy_script(path_output)

    logger.info(f'\nBIDS conversion completed at {datetime.datetime.now()}')
    logger.info(f'Processed {len(participants)} participants')
    logger.info(f'Successfully processed {processed_images} NIfTI images')
    logger.info(f'Successfully processed {processed_jsons} JSON files')


if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args()

    # Default paths from your provided tree.txt example
    path_dataset = "/Users/nathan/data/amos22/amos22"  # os.path.abspath(args.path_dataset)
    path_output = "/Users/nathan/data/amos22/abdominal-amos22"  # os.path.abspath(args.path_output)
    main(path_dataset, path_output)
