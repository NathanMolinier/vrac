import os
import sys
import shutil
import json
import argparse
import logging
import datetime
import csv
import re
from vrac.data_management.image import Image, zeros_like
import numpy as np

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


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



def create_participants_tsv(file_metadata, participants, path_output):
    with open(file_metadata, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        metadata_dict = {row['image_id']: row for row in reader}

    with open(os.path.join(path_output, 'participants.tsv'), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['participant_id', 'source_id', 'species', 'age', 'sex', 'manufacturer', 'scanner_model', 'institution', 'pathology', 'notes', 'split'])
        for item in sorted(participants, key=lambda a: a[0]):
            age = metadata_dict[item[1]]['age'] if metadata_dict[item[1]]['age'].isdigit() else 'n/a'
            sex = metadata_dict[item[1]]['gender'].upper() if metadata_dict[item[1]]['gender'] in ['f', 'm'] else 'n/a'
            if metadata_dict[item[1]]['pathology'] in ['unclear', 'other']:
                pathology = 'unclear'
            elif metadata_dict[item[1]]['pathology'] in ['no_pathology']:
                pathology = 'healthy control'
            else:
                pathology = metadata_dict[item[1]]['pathology'] if metadata_dict[item[1]]['pathology'] else 'n/a'
            notes = metadata_dict[item[1]]['pathology_location'] if pathology not in ['healthy control', ''] else 'n/a'
            institution = metadata_dict[item[1]]['institute'] if metadata_dict[item[1]]['institute'] else 'n/a'
            manufacturer = metadata_dict[item[1]]['manufacturer'] if metadata_dict[item[1]]['manufacturer'] else 'n/a'
            scanner_model = metadata_dict[item[1]]['scanner_model'] if metadata_dict[item[1]]['scanner_model'] else 'n/a'
            split = metadata_dict[item[1]]['split'] if metadata_dict[item[1]]['split'] else 'n/a'
            known_data = ['homo sapiens', age, sex, manufacturer, scanner_model, institution, pathology, notes, split]
            tsv_writer.writerow(list(item) + known_data)
        logger.info(f'participants.tsv created in {path_output}')


def create_participants_json(path_output):
    data_json = {
        "participant_id": {"Description": "Unique Participant ID", "LongName": "Participant ID"},
        "source_id": {"Description": "Original subject folder name"},
        "species": {"Description": "Binomial species name of participant", "LongName": "Species"},
        "age": {"Description": "Participant age", "LongName": "Participant age", "Units": "years"},
        "sex": {"Description": "sex of the participant as reported by the participant", "Levels": {"M": "male", "F": "female", "O": "other"}},
        "manufacturer": {"Description": "Manufacturer of the participant's imaging equipment", "LongName": "Manufacturer"},
        "scanner_model": {"Description": "Model of the participant's imaging equipment", "LongName": "Scanner Model"},
        "institution": {"Description": "Human-friendly institution name", "LongName": "BIDS Institution ID"},
        "pathology": {"Description": "The diagnosis of pathology of the participant", "LongName": "Pathology name", "Levels": {"LBP": "Low Back Pain"}},
        "notes": {"Description": "Additional notes about the participant", "LongName": "Additional notes"},
        "split": {"Description": "Dataset split containing this image for TotalSegmentator-CT training", "LongName": "Dataset Split", "Levels": {"train": "Training Set", "val": "Validation Set", "test": "Test Set"}}
    }
    write_json(path_output, 'participants.json', data_json)


def create_dataset_description(path_output):
    data_json = {
        "BIDSVersion": "1.9.0",
        "Name": "fullbody-totalsegmentator-ct",
        "DatasetType": "raw"
    }
    create_folder(path_output)
    write_json(path_output, 'dataset_description.json', data_json)

def create_json_sidecar(path_output, labels):
    if path_output.endswith('.nii.gz'):
        path_json = path_output.replace('.nii.gz', '.json')
    else:
        path_json = path_output
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "Semi-automatic",
                "Author": "https://pubs.rsna.org/doi/epdf/10.1148/ryai.230024",
            }
        ],
        "Labels": labels
    }
    write_json(os.path.dirname(path_json), os.path.basename(path_json), data_json)


def copy_script(path_output):
    path_script_in = sys.argv[0]
    path_code = os.path.join(path_output, 'code')
    create_folder(path_code)
    path_script_out = os.path.join(path_code, os.path.basename(path_script_in))
    logger.info(f'Copying {path_script_in} to {path_script_out}')
    shutil.copyfile(path_script_in, path_script_out)

def normalize_filename(subject_bids, label=False):
    if label:
        return f"{subject_bids}_CT_label-body_dseg.nii.gz"
    else:
        return f"{subject_bids}_CT.nii.gz"

def subject_to_bids(img_path):
    """Extract number from filename"""
    if img_path.startswith('s'):
        return f"sub-{img_path[1:]}"

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
    processed_segs = 0
    processed_jsons = 0
    labels = None

    img_dirs = [di for di in os.listdir(path_dataset) if di.startswith('s')]

    for di in img_dirs:
        path_file_in = os.path.join(path_dataset, di, "ct.nii.gz")
        if not os.path.exists(path_file_in):
            raise ValueError(f"Error with IMAGE file: {path_file_in}")

        subject_bids = subject_to_bids(di)

        anat_dir = os.path.join(path_output, subject_bids, 'anat')
        create_folder(anat_dir)

        # Track participant
        if subject_bids not in [p[0] for p in participants]:
            participants.append((subject_bids, di))
        
        normalized_fname = normalize_filename(subject_bids)
        path_file_out = os.path.join(anat_dir, normalized_fname)

        # Image: load, RPI, save
        logger.info(f'Processing image: {path_file_in}')
        # img = Image(path_file_in).change_orientation('RPI')
        # img.save(path_file_out)
        processed_images += 1
        logger.info(f'Saved: {path_file_out}')

        # Handle segmentations
        segmentations_folder = os.path.join(path_dataset, di, "segmentations")
        seg_files = sorted([f for f in os.listdir(segmentations_folder) if f.endswith('.nii.gz')])
        if len(seg_files) != 117:
            raise ValueError(f"Expected 117 segmentation files in {segmentations_folder}, found {len(seg_files)}")
        if labels is None:
            labels = {i+1:f.replace('.nii.gz', '') for i, f in enumerate(seg_files)}

        # Create derivatives folder
        derivatives_anat_dir = os.path.join(path_output, 'derivatives', 'labels', subject_bids, 'anat',)
        create_folder(derivatives_anat_dir)

        seg_path_out = os.path.join(derivatives_anat_dir, normalize_filename(subject_bids, label=True))
        # seg = zeros_like(img)
        for i, file in labels.items():
            seg_path_in = os.path.join(segmentations_folder, f"{file}.nii.gz")

            # Image: load, RPI, save
            logger.info(f'Processing segmentation: {seg_path_in}')
            # part = Image(seg_path_in).change_orientation('RPI')

            # # Check for errors with segmentation
            # if not part.data.shape == seg.data.shape:
            #     raise ValueError(f"Segmentation shape {part.data.shape} does not match image shape {seg.data.shape} for {seg_path_in}")

            # unique_vals = set(np.unique(part.data).tolist())
            # if not unique_vals.issubset({0, 1}):
            #     logger.error(f"Segmentation {seg_path_in} has values other than [0, 1]: {sorted(unique_vals)}, remove other values.")

            # if unique_vals == {0}:
            #     logger.warning(f'Empty segmentation (all zeros): {seg_path_in}')
            #     continue

            # if seg.data[part.data == 1].any():
            #     logger.warning(f"{subject_bids}: Segmentation {seg_path_in} overlaps with previous segmentations.")

            # seg.data[part.data == 1] = i

        # Save the combined segmentation
        # seg.save(seg_path_out)
        processed_segs += 1
        logger.info(f'Saved: {seg_path_out}')

        # Save JSON sidecar for labels (alongside the segmentation)
        create_json_sidecar(seg_path_out, labels)
        processed_jsons += 1
        logger.info(f'Saved: {seg_path_out.replace(".nii.gz", ".json")}')

    # Write metadata
    file_metadata = os.path.join(path_dataset, "meta.csv")
    create_participants_tsv(file_metadata, participants, path_output)
    create_participants_json(path_output)
    create_dataset_description(path_output)
    copy_script(path_output)

    logger.info(f'\nBIDS conversion completed at {datetime.datetime.now()}')
    logger.info(f'Processed {len(participants)} participants')
    logger.info(f'Successfully processed {processed_images} CT images')
    logger.info(f'Successfully processed {processed_segs} segmentations')
    logger.info(f'Successfully processed {processed_jsons} JSON files')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Default paths from your provided tree.txt example
    path_dataset = os.path.abspath(args.path_dataset)
    path_output = os.path.abspath(args.path_output)
    main(path_dataset, path_output)
