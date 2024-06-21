import os
import sys
import shutil
import json
import argparse
import logging
import datetime
import csv
import subprocess

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

   
def copy_nii(path_file_in, path_file_out):
    """
    Copy nii file from the input dataset to the output dataset
    :param path_file_in: path to nii file in the input dataset
    :param path_file_out: path to nii file in the output dataset
    """
    shutil.copy(path_file_in, path_file_out)
    logger.info(f'Copying: {path_file_in} --> {path_file_out}')
    gzip_nii(path_file_out)


def copy(path_file_in, path_file_out):
    """
    Copy file from the input dataset to the output dataset
    :param path_file_in: path to file in the input dataset
    :param path_file_out: path to file in the output dataset
    """
    shutil.copy(path_file_in, path_file_out)
    logger.info(f'Copying: {path_file_in} --> {path_file_out}')


def gzip_nii(path_nii):
    """
    Gzip nii file
    :param path_nii: path to the nii file
    :return:
    """
    # Check if file is gzipped
    if not path_nii.endswith('.gz'):
        path_gz = path_nii + '.gz'
        logger.info(f'Gzipping {path_nii} to {path_gz}')
        os.system(f'gzip -f {path_nii}')


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
    :param participants_tsv_list: list containing ['participant_id', 'source_id']
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
    data_json = {
        "BIDSVersion": "1.9.0",
        "Name": dataset_name,
        "DatasetType": "raw" if not derivative else "derivative"
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
    path_output = os.path.abspath(args.path_output)
    dataset_name = os.path.normpath(os.path.basename(args.path_output))

    # Check if input path is valid
    if not os.path.isdir(path_dataset):
        print(f'ERROR - {path_dataset} does not exist.')
        sys.exit()
    
    # Create output folder if it does not exist
    if not os.path.isdir(path_output):
        os.makedirs(path_output, exist_ok=True)

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
    if os.path.exists(os.path.join(path_output, 'participants.tsv')):
        with open(os.path.join(path_output, 'participants.tsv')) as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')
            sub_dict_tsv = dict(reader)
            del sub_dict_tsv['participant_id']
    else:
        sub_dict_tsv = dict()
    
    for dir in os.listdir(path_dataset):
        path_input = os.path.join(path_dataset, dir)
        if os.path.isdir(path_input):
            sub_list = sorted([d for d in os.listdir(path_input) if d.startswith('sub')])
            for sub in sub_list:
                for file in os.listdir(os.path.join(path_input, sub)):
                    if file.endswith('.nii.gz'):                    
                        # Add subject name to participant.tsv 
                        if sub not in sub_dict_tsv.keys(): # Add only one time each subject into the participant.csv
                            # Aggregate subjects for participants.tsv
                            sub_dict_tsv[sub] = sub

                        # Path input image
                        path_file_in = os.path.join(path_input, sub, file)

                        # Construct path for the output IMAGE
                        path_subject_folder_out = os.path.join(path_output, sub, 'anat')
                        create_subject_folder_if_not_exists(path_subject_folder_out)
                        path_file_out = os.path.join(path_subject_folder_out, file)

                        # Copy nii
                        copy_nii(path_file_in, path_file_out)

                        # Reorient image to RPI
                        subprocess.check_call([
                            "sct_image",
                            "-i", path_file_out,
                            "-setorient", "RPI"
                        ])

                        # Copy json sidecar
                        copy(path_file_in.replace('.nii.gz', '.json'), path_file_out.replace('.nii.gz', '.json'))

    participants_tsv_list = list(sub_dict_tsv.items())

    create_participants_tsv(participants_tsv_list, path_output)
    create_participants_json(path_output)
    create_dataset_description(path_output, dataset_name=dataset_name, derivative=False)
    copy_script(path_output)

if __name__ == "__main__":
    main()          