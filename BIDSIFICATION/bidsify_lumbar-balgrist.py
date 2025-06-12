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


def copy(path_file_in, path_file_out):
    """
    Copy file from the input dataset to the output dataset
    :param path_file_in: path to file in the input dataset
    :param path_file_out: path to file in the output dataset
    """
    shutil.copy(path_file_in, path_file_out)
    logger.info(f'Copying: {path_file_in} --> {path_file_out}')


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


def create_dataset_description(path_output):
    """
    Create dataset_description.json file
    :param path_output: path to the output BIDS folder
    :return:
    """
    data_json = {
        "BIDSVersion": "BIDS 1.9.0",
        "Name": "lumbar-balgrist",
        "DatasetType": "raw"
    }
    if not os.path.isdir(path_output):
        os.makedirs(path_output, exist_ok=True)
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
    sub_dict_tsv = dict()
    
    # List excluded data
    exclude = []
    for dir in os.listdir(path_dataset):
        path_input = os.path.join(path_dataset, dir)
        if os.path.isdir(path_input):
            num = dir.replace('Anonym_Patient', '')
            subject_name_bids = f"sub-{num}"
            sub_files = []
            for file in os.listdir(path_input):
                if file.endswith('.nii.gz'):
                    if not '_ROI' in file:
                        sub_files.append(file)
                        # Add subject name to participant.tsv 
                        if subject_name_bids not in sub_dict_tsv.keys(): # Add only one time each subject into the participant.csv
                            # Aggregate subjects for participants.tsv
                            sub_dict_tsv[subject_name_bids] = dir

                        # Path input image
                        path_file_in = os.path.join(path_input, file)

                        # Fetch image acquisition type
                        acq_entity = ''
                        chunk_entity = ''
                        if '_sag_' in file or '_SAG_' in file or '_Sag_' in file or '-SAG_' in file:
                            acq_entity = 'acq-sag'
                        elif '_tra_' in file or '_AX_' in file or '_ax_' in file or '_Ax_' in file or '_TRA_' in file:
                            acq_entity = 'acq-ax'
                            if file.split('_')[-1].startswith('i0'):
                                chunk_num = file.split('_i')[-1].split('.')[0]
                            else:
                                chunk_num = '00001'
                            chunk_entity = f'chunk-{chunk_num}'
                        
                        if not acq_entity and file.split('_')[-1].startswith('i0'):
                            chunk_num = file.split('_i')[-1].split('.')[0]
                            chunk_entity = f'chunk-{chunk_num}'
                            acq_entity = 'acq-ax'

                        # Fetch contrast
                        cont = ''
                        if '_t1_' in file or '_T1_' in file or '_T1W_' in file or '_eT1_' in file:
                            cont = 'T1w'
                        elif '_t2_' in file or '_T2_' in file or '_T2W_' in file or '_T2-' in file or '_sT2W_' in file or '_eT2_' in file or '_eT2W_' in file:
                            cont = 'T2w'
                        
                        # Load image and reorient
                        img = Image(path_file_in).change_orientation('RPI')

                        # If no acq_entity check resolution
                        if not acq_entity:
                            x, y, z, t, xr, yr, zr, tr = img.dim
                            if xr > 5*yr and xr > 5*zr: # Higher R-L pixdim --> lower sagittal resolution, probably saggital
                                acq_entity = 'acq-sag'
                        
                        # Check acq entity and contrast
                        if not acq_entity:
                            raise ValueError(f'Missing acq entity with filename: {file}')
                        if not cont:
                            raise ValueError(f'Missing contrast with filename: {file}')
                        
                        # Create image filename
                        if acq_entity == 'acq-ax':
                            img_filename = "_".join([subject_name_bids, acq_entity, chunk_entity, cont])
                        else:
                            img_filename = "_".join([subject_name_bids, acq_entity, cont])
                        
                        # Construct path for the output IMAGE
                        path_subject_folder_out = os.path.join(path_output, subject_name_bids, 'anat')
                        create_subject_folder_if_not_exists(path_subject_folder_out)
                        filename_out = img_filename + '.nii.gz'
                        path_file_out = os.path.join(path_subject_folder_out, filename_out)

                        # Check if multiple files are projected onto the same output file
                        if os.path.exists(path_file_out):
                            raise ValueError(f'Multiple files have the same name for subject {dir}: see {path_file_out}')

                        # Save file nifti
                        img.save(path_file_out)

                        # Copy JSON 
                        copy(path_file_in.replace('.nii.gz', '.json'), path_file_out.replace('.nii.gz', '.json'))
                    else:
                        exclude.append(file)
    
    participants_tsv_list = list(sub_dict_tsv.items())                 

    create_participants_tsv(participants_tsv_list, path_output)
    create_participants_json(path_output)
    create_dataset_description(path_output)
    copy_script(path_output)

if __name__ == "__main__":
    main()          