import os
import sys
import shutil
import json
import argparse
import logging
import datetime
import csv
from vrac.data_management.image import Image

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

def get_parser():
    parser = argparse.ArgumentParser(description='Convert dataset to BIDS format.')
    parser.add_argument("-i", "--path-dataset",
                        help="Path to the non-BIDS dataset.",
                        required=True)
    parser.add_argument("-o", "--path-output",
                        help="Path to the output folder where the BIDS dataset will be stored.",
                        required=True)
    return parser


def create_subject_folder_if_not_exists(path_subject_folder_out):
    """
    Check if subject's folder exists in the output dataset, if not, create it
    :param path_subject_folder_out: path to subject's folder in the output dataset
    """
    if not os.path.isdir(path_subject_folder_out):
        os.makedirs(path_subject_folder_out)
        logger.info(f'Creating directory: {path_subject_folder_out}')


def edit_participants_tsv(participants_tsv_list, path_output):
    """
    Write participants.tsv file
    :param participants_tsv_list: list containing [subject_out, pathology_out, subject_in, centre_in, centre_out],
    example:[sub-torontoDCM001, DCM, 001, 01, toronto]
    :param path_output: path to the output BIDS folder
    :return:
    """
    if os.path.exists(os.path.join(path_output, 'participants.tsv')):
        with open(os.path.join(path_output, 'participants.tsv'), 'r') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            list_subjects = [row for row in tsv_reader][1:]  # Skip header
    
    with open(os.path.join(path_output, 'participants.tsv'), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['participant_id', 'source_id', 'species', 'age', 'sex', 'pathology', 'institution', 'notes'])
        
        # Species
        species = ['homo sapiens']

        # Extra info
        extra_data = ['n/a', 'n/a', 'n/a', 'n/a', 'n/a']

        # Add new subjects
        for participant in participants_tsv_list:
            list_subjects.append([participant[0], participant[1]] + species + extra_data)

        # Add rows to tsv file
        list_subjects = sorted(list_subjects, key=lambda a : a[0])
        for item in list_subjects:
            tsv_writer.writerow(item)
        logger.info(f'participants.tsv created in {path_output}')


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
    path_script_out = os.path.join(path_code, sys.argv[0].split(sep='/')[-1])
    logger.info(f'Copying {path_script_in} to {path_script_out}')
    shutil.copyfile(path_script_in, path_script_out)


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Make sure that input args are absolute paths
    path_dataset = os.path.abspath(args.path_dataset)
    path_output = os.path.abspath(args.path_output)

    # Check if input path is valid
    if not os.path.isdir(path_dataset):
        print(f'ERROR - {path_dataset} does not exist.')
        sys.exit()

    FNAME_LOG = os.path.join(path_output, 'bids_conversion.log')
    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    
    fh = logging.FileHandler(os.path.join(os.path.abspath(os.curdir), FNAME_LOG))
    logging.root.addHandler(fh)
    logger.info("INFO: log file will be saved to {}".format(FNAME_LOG))

    # Print current time and date to log file
    logger.info('\nAnalysis started at {}'.format(datetime.datetime.now()))
    
    # Initialize dict for participants.tsv
    sub_dict_tsv = dict()
    for file in os.listdir(path_dataset):
        if file.endswith('.nii.gz'):
            path_input = os.path.join(path_dataset, file)
            old_filename = os.path.basename(path_input)
            old_name = "_".join(old_filename.split('_')[0:2])
            bids_filename = old_filename.replace('RESTORE_','')
            session = bids_filename.split('_ses-')[1].split('_')[0]
            subject_name_bids = bids_filename.split('_')[0]
            
            # Add subject name to participant.tsv 
            if subject_name_bids not in sub_dict_tsv.keys(): # Add only one time each subject into the participant.csv
                # Aggregate subjects for participants.tsv
                sub_dict_tsv[subject_name_bids] = old_name

            # Path input image
            path_file_in = os.path.join(path_input)
            
            # Load image and reorient
            img = Image(path_file_in).change_orientation('RPI')
            
            # Construct path for the output IMAGE
            path_subject_folder_out = os.path.join(path_output, subject_name_bids, f'ses-{session}', 'anat')
            create_subject_folder_if_not_exists(path_subject_folder_out)

            # Save file nifti
            path_file_out = os.path.join(path_subject_folder_out, bids_filename)
            img.save(path_file_out)
    
    participants_tsv_list = list(sub_dict_tsv.items())                 

    edit_participants_tsv(participants_tsv_list, path_output)
    copy_script(path_output)

if __name__ == "__main__":
    main()          