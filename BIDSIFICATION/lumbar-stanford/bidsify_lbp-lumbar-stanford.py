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
        extra_data = ['n/a', 'n/a', 'LBP', 'Stanford', 'n/a']
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
        "Name": "lbp-lumbar-stanford",
        "DatasetType": "raw"
    }
    create_folder(path_output)
    write_json(path_output, 'dataset_description.json', data_json)


def copy_script(path_output):
    path_script_in = sys.argv[0]
    path_code = os.path.join(path_output, 'code')
    create_folder(path_code)
    path_script_out = os.path.join(path_code, os.path.basename(path_script_in))
    logger.info(f'Copying {path_script_in} to {path_script_out}')
    shutil.copyfile(path_script_in, path_script_out)


def subject_to_bids(subject_dir_name):
    """Map numeric subject folder (e.g., '51') to BIDS ID 'sub-051'."""
    if not subject_dir_name.isdigit():
        # Fallback: sanitize
        slug = re.sub(r'[^a-zA-Z0-9]', '', subject_dir_name)
        return f"sub-{slug}"
    return f"sub-{int(subject_dir_name):03d}"


def normalize_filename(fname, subject_bids, session_bids):
    """
    Normalize Stanford filenames into BIDS order and semantics.
    Handles known patterns from tree.txt:
    - 3D_SAG_T2W_Cube_FS.(nii|json) -> acq-Sag3dCubefs_T2w
    - FAT_3D_AX_IDEAL_IQ.(nii|json) -> acq-Ax3dFat_T2w
    - WATER_3D_AX_IDEAL_IQ.(nii|json) -> acq-Ax3dWater_T2w
    """
    base = os.path.splitext(fname)[0]
    ext = os.path.splitext(fname)[1]
    # Preserve .nii.gz if already gz
    if ext == '.gz':
        base2, ext2 = os.path.splitext(base)
        base = base2
        ext = ext2 + '.gz'

    acq = None
    contrast = None

    name = base.lower()
    # Identify patterns
    if '3d_sag_t2w_cube_fs' in name or '3dsagt2wcubefs' in name:
        acq = 'Sag3dCubeFS'
        contrast = 'T2w'
    elif 'fat_3d_ax_ideal_iq' in name or 'fat3daxidealiq' in name:
        acq = 'Ax3dFat'
        contrast = 'T2w'
        contrast = 'T2w'
    elif 'water_3d_ax_ideal_iq' in name or 'water3daxidealiq' in name:
        acq = 'Ax3dWater'
        contrast = 'T2w'
    else:
        # Fallback: try to infer axial/sag and T1/T2
        acq = 'axial' if 'ax' in name else 'sagittal' if 'sag' in name else 'axial'
        contrast = 'T2w' if 't2' in name else 'T1w' if 't1' in name else 'T2w'

    parts = [subject_bids, session_bids]
    if acq:
        parts.append(f'acq-{acq}')
    parts.append(contrast)

    out_fname = '_'.join(parts)
    # Use .nii.gz for images, keep .json for sidecars
    if ext.endswith('.nii'):
        ext = '.nii.gz'
    return out_fname + ext


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

    # Each immediate child under dataset root is a subject folder (e.g., '51', '56', '81')
    for subject_dir in sorted(os.listdir(path_dataset)):
        subj_path = os.path.join(path_dataset, subject_dir)
        if not os.path.isdir(subj_path):
            continue

        subject_bids = subject_to_bids(subject_dir)
        session_bids = 'ses-01'

        anat_dir = os.path.join(path_output, subject_bids, session_bids, 'anat')
        create_folder(anat_dir)

        # Track participant
        if subject_bids not in [p[0] for p in participants]:
            participants.append((subject_bids, subject_dir))

        # Process only top-level NIfTI/JSON inside subject folder; skip DICOM subfolders
        for fname in os.listdir(subj_path):
            path_file_in = os.path.join(subj_path, fname)
            if os.path.isdir(path_file_in):
                # Known DICOM series folders: skip
                continue
            if not (fname.endswith('.nii') or fname.endswith('.nii.gz') or fname.endswith('.json')):
                continue

            try:
                normalized_fname = normalize_filename(fname, subject_bids, session_bids)
                path_file_out = os.path.join(anat_dir, normalized_fname)

                if fname.endswith('.json'):
                    logger.info(f'Copying JSON: {path_file_in} -> {path_file_out}')
                    shutil.copy(path_file_in, path_file_out)
                    processed_jsons += 1
                else:
                    # Image: load, RPI, save
                    logger.info(f'Processing image: {path_file_in}')
                    img = Image(path_file_in).change_orientation('RPI')
                    img.save(path_file_out)
                    processed_images += 1
                    logger.info(f'Saved: {path_file_out}')
            except Exception as e:
                logger.error(f"ERROR - Failed to process {path_file_in}: {e}")

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
    path_dataset = "/Users/nathan/Desktop/SpineMRI_Pfirrmann_AG25"  # os.path.abspath(args.path_dataset)
    path_output = "/Users/nathan/data/lbp-lumbar-stanford"  # os.path.abspath(args.path_output)
    main(path_dataset, path_output)
