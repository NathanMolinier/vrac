import os
import sys
import shutil
import json
import argparse
import logging
import datetime
import csv
from vrac.data_management.image import Image
from progress.bar import Bar

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

def get_parser():
    parser = argparse.ArgumentParser(description='Convert dataset to BIDS format.')
    parser.add_argument("-i", "--path-dataset",
                        help="Path to the raw dataset.",
                        required=True)
    parser.add_argument("-o", "--out-bids",
                        help="Path to the BIDS dataset.",
                        required=True)
    return parser


def update_json_file(path_in, path_out):
    """
    Create a json sidecar file
    :param path_in: path to the input file
    :param path_out: path to the output file
    """
    # Read json file and create a dictionary
    with open(path_in, "r") as file:
        json_info = json.load(file)

    top_dict = {}
    if not "SpatialReference" in json_info.keys():
        top_dict["SpatialReference"] = "orig"
    else:
        top_dict["SpatialReference"] = json_info["SpatialReference"]
        json_info.pop("SpatialReference")

    if not "GeneratedBy" in json_info.keys():
        d = {"Name": "Manual"}
        if 'info' in json_info.keys():
            if 'software' in json_info['info'].keys():
                d["Sofware"] = json_info['info']['software']
            if 'version' in json_info['info'].keys():
                d["Version"] = json_info['info']['version']
            json_info.pop("info")
        if 'segmentation_quality' in json_info.keys():
            if 'vert_version' in json_info['segmentation_quality'].keys():
                d["Author"] = json_info['segmentation_quality']['vert_version']
            if 'vert_quality' in json_info['segmentation_quality'].keys():
                d["Quality"] = json_info['segmentation_quality']['vert_quality']
        else:
            d["Description"]= "Segmentation manually generated for the verse challenge"
        top_dict["GeneratedBy"] = [d]
    else:
        raise ValueError('Unexpected field')

    json_data = {**top_dict, **json_info}
    with open(path_out, 'w') as f:
        json.dump(json_data, f, indent=4)
        

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
    with open(os.path.join(path_output, 'participants.json'), 'w') as json_file:
        json.dump(data_json, json_file, indent=4)
        # Add last newline
        json_file.write("\n")
        logger.info(f'participants.json created in {path_output}')
    

def create_dataset_description(path_output):
    """
    Create dataset_description.json file
    :param path_output: path to the output BIDS folder
    :return:
    """
    data_json = {
        "BIDSVersion": "BIDS 1.9.0",
        "Name": f"verse-challenge-{os.path.basename(path_output).split('-')[-2]}-ct",
        "DatasetType": "raw",
        "Authors": ["Nathan Molinier"]
    }

    with open(os.path.join(path_output, 'dataset_description.json'), 'w') as json_file:
        json.dump(data_json, json_file, indent=4)
        # Add last newline
        json_file.write("\n")
        logger.info(f'dataset_description.json created in {path_output}')


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
    out_folder = os.path.abspath(args.out_bids)

    # Check if input path is valid
    if not os.path.isdir(path_dataset):
        print(f'ERROR - {path_dataset} does not exist.')
        sys.exit()

    FNAME_LOG = os.path.join(out_folder, 'bids_conversion.log')
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

    # Generate dataset description
    create_dataset_description(out_folder)
    
    # Loop through subjects
    raw_folder = os.path.join(path_dataset, 'rawdata')
    derivatives = os.path.join(path_dataset, 'derivatives')
    missing_files = []
    error_files = []

    # Init progression bar
    subs_folder = sorted(os.listdir(raw_folder))
    bar = Bar(f'Load data with pre-processing', max=len(subs_folder))

    for sub in subs_folder:
        if sub.startswith('sub'):
            sub_folder = os.path.join(raw_folder, sub)
            for file_name in os.listdir(sub_folder):
                if file_name.endswith('.nii.gz'):
                    if 'split' in file_name:
                        file_nb = int(file_name.split('_')[1].split('verse')[-1])
                        list_nb = sorted([int(name.split('_')[1].split('verse')[-1]) for name in os.listdir(sub_folder) if name.endswith('.nii.gz')])
                        chunk_index = list_nb.index(file_nb)+1
                    else:
                        chunk_index = 1

                    # Get image path
                    img_name = file_name
                    img_nii = os.path.join(sub_folder, img_name)
                    img_json = os.path.join(raw_folder, sub, img_name.replace('.nii.gz', '.json'))

                    # Get segmentation path
                    seg_nii = os.path.join(derivatives, sub, f'{img_name.replace('_ct.nii.gz','')}_seg-vert_msk.nii.gz')
                    seg_json = os.path.join(derivatives, sub, f'{img_name.replace('_ct.nii.gz','')}_labels.json')

                    if os.path.exists(img_nii) and os.path.exists(img_json) and os.path.exists(seg_nii) and os.path.exists(seg_json):
                        # Check image resolution and change orientation
                        img = Image(img_nii).change_orientation('RPI')
                        seg = Image(seg_nii).change_orientation('RPI')

                        if img.dim[:3] != seg.dim[:3]:
                            error_files.append(sub)
                        else:
                            # Fetch number to the chunk
                            chunk = f'_chunk-{chunk_index}'

                            # Rename files
                            out_img_name = f'{sub}{chunk}_CT.nii.gz'
                            out_img_nii = os.path.join(out_folder, sub, 'anat', out_img_name)

                            out_seg_name = f'{sub}{chunk}_CT_label-spine_dseg.nii.gz'
                            out_seg_nii = os.path.join(out_folder, 'derivatives/labels', sub, 'anat', out_seg_name)

                            # Create image directories
                            if not os.path.exists(os.path.dirname(out_img_nii)):
                                os.makedirs(os.path.dirname(out_img_nii))
                            
                            # Create segmentation directories
                            if not os.path.exists(os.path.dirname(out_seg_nii)):
                                os.makedirs(os.path.dirname(out_seg_nii))
                            
                            # Save JSON files
                            shutil.copyfile(img_json, out_img_nii.replace('.nii.gz', '.json'))
                            update_json_file(seg_json, out_seg_nii.replace('.nii.gz', '.json'))
                            
                            # Save output niftii files
                            img.save(out_img_nii)
                            seg.save(out_seg_nii)

                            if sub not in sub_dict_tsv.keys(): # Add only one time each subject into the participant.csv
                                # Aggregate subjects for participants.tsv
                                orig_name = sub.split('-')[-1]
                                sub_dict_tsv[sub] = orig_name

                    else:
                        missing_files.append(sub)
        # Plot progress
        bar.suffix  = f'{subs_folder.index(sub)+1}/{len(subs_folder)}'
        bar.next()
    bar.finish()

    print("missing files:\n" + '\n'.join(sorted(missing_files)))
    print("error files:\n" + '\n'.join(sorted(error_files)))

    participants_tsv_list = list(sub_dict_tsv.items())                 

    create_participants_tsv(participants_tsv_list, out_folder)
    create_participants_json(out_folder)
    copy_script(out_folder)

if __name__ == "__main__":
    main()          