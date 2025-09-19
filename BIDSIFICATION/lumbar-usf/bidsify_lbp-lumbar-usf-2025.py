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
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

def get_parser():
    parser = argparse.ArgumentParser(description='Convert lbp-lumbar-usf-2025 dataset to BIDS format.')
    parser.add_argument("-i", "--path-dataset",
                        help="Path to the sourcedata folder containing the non-BIDS dataset.",
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


def copy_file(path_file_in, path_file_out):
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
        

def read_demographics(demographics_file):
    """
    Read demographics CSV file and return a dictionary with subject information
    :param demographics_file: path to the demographics CSV file
    :return: dictionary with subject info keyed by subject ID and session
    """
    demographics = {}
    
    if not os.path.exists(demographics_file):
        logger.warning(f"Demographics file not found: {demographics_file}")
        return demographics
    
    try:
        # Open with utf-8-sig to handle BOM character
        with open(demographics_file, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Handle the BOM character in the first column name
                subject_id_key = 'Subject ID'
                if '\ufeffSubject ID' in row:
                    subject_id_key = '\ufeffSubject ID'
                elif 'Subject ID' in row:
                    subject_id_key = 'Subject ID'
                else:
                    # Find the key that contains 'Subject ID'
                    for key in row.keys():
                        if 'Subject ID' in key:
                            subject_id_key = key
                            break
                
                subject_id = row[subject_id_key].strip()
                session = row['Pre/Post'].strip()
                
                # Skip empty rows
                if not subject_id or not session:
                    continue
                
                # Handle special cases like Post1, Post2
                if session.startswith('Post'):
                    if session == 'Post':
                        session = 'Post'
                    else:
                        session = 'Post'  # Treat Post1, Post2 as Post for BIDS
                
                key = f"{subject_id}_{session}"
                demographics[key] = {
                    'sex': row['Sex'].strip() if row['Sex'].strip() else 'n/a',
                    'age': row['Age'].strip() if row['Age'].strip() else 'n/a',
                    'level_dx': row['Level Dx'].strip() if row['Level Dx'].strip() else 'n/a',
                    'dx_code': row['Dx Code\n(ICD-10 / ICD-9)'].strip() if row['Dx Code\n(ICD-10 / ICD-9)'].strip() else 'n/a',
                    'level_treated': row['Level Treated\n(e.g. L4, L5)'].strip() if row['Level Treated\n(e.g. L4, L5)'].strip() else 'n/a'
                }
        
        logger.info(f"Successfully read demographics for {len(demographics)} subject-session combinations")
        
    except Exception as e:
        logger.error(f"Error reading demographics file: {str(e)}")
    
    return demographics


def create_participants_tsv(participants_tsv_list, demographics, path_output):
    """
    Write participants.tsv file with demographics information
    :param participants_tsv_list: list containing [subject_out, source_id],
    example: [sub-nMRI001, nMRI001]
    :param demographics: dictionary with demographics information
    :param path_output: path to the output BIDS folder
    :return:
    """
    with open(os.path.join(path_output, 'participants.tsv'), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['participant_id', 'source_id', 'species', 'age', 'sex', 'pathology', 
                            'institution', 'level_dx', 'dx_code', 'level_treated', 'notes'])
        
        # Default values
        species = 'homo sapiens'
        pathology = 'LBP'  # Low Back Pain
        institution = 'USF'  # University of South Florida
        notes = 'n/a'

        # Add rows to tsv file
        participants_tsv_list = sorted(participants_tsv_list, key=lambda a : a[0])
        for item in participants_tsv_list:
            subject_bids = item[0]  # e.g., sub-nMRI001
            source_id = item[1]     # e.g., nMRI001
            
            # Extract numeric ID for demographics lookup
            numeric_id = source_id.replace('nMRI', '').zfill(3)  # e.g., 001
            
            # Try to get demographics for this subject (try both Pre and Post sessions)
            demo_key_pre = f"{numeric_id.lstrip('0')}_Pre"  # Remove leading zeros for lookup
            demo_key_post = f"{numeric_id.lstrip('0')}_Post"
            
            demo_info = demographics.get(demo_key_pre) or demographics.get(demo_key_post)
            
            if demo_info:
                age = demo_info['age']
                sex = demo_info['sex']
                level_dx = demo_info['level_dx']
                dx_code = demo_info['dx_code']
                level_treated = demo_info['level_treated']
            else:
                age = 'n/a'
                sex = 'n/a'
                level_dx = 'n/a'
                dx_code = 'n/a'
                level_treated = 'n/a'
                logger.warning(f"No demographics found for subject {subject_bids}")
            
            tsv_writer.writerow([subject_bids, source_id, species, age, sex, pathology, 
                               institution, level_dx, dx_code, level_treated, notes])
        
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
                "NRC": "Nerve Root Compression",
                "DCM": "Degenerative Cervical Myelopathy (synonymous with CSM - Cervical Spondylotic Myelopathy)",
                "MildCompression": "Asymptomatic cord compression, without myelopathy",
                "MS": "Multiple Sclerosis",
                "SCI": "Traumatic Spinal Cord Injury",
                "LBP": "Low Back Pain"
            }
        },
        "institution": {
            "Description": "Human-friendly institution name",
            "LongName": "BIDS Institution ID"
        },
        "level_dx": {
            "Description": "Spinal level where diagnosis was made",
            "LongName": "Diagnosis Level"
        },
        "dx_code": {
            "Description": "ICD-10 or ICD-9 diagnosis code",
            "LongName": "Diagnosis Code"
        },
        "level_treated": {
            "Description": "Spinal level that was treated",
            "LongName": "Treatment Level"
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
        "BIDSVersion": "1.9.0",
        "Name": "lbp-lumbar-usf-2025",
        "DatasetType": "raw",
        "License": "n/a",
        "Authors": ["n/a"]
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
    path_script_out = os.path.join(path_code, os.path.basename(path_script_in))
    logger.info(f'Copying {path_script_in} to {path_script_out}')
    shutil.copyfile(path_script_in, path_script_out)


def get_contrast_from_json(json_file_path):
    """
    Extract contrast information from JSON sidecar file using PulseSequenceName
    :param json_file_path: path to the JSON sidecar file
    :return: tuple of (contrast_suffix, acquisition_info) or (None, None) if not found
    """
    if not os.path.exists(json_file_path):
        return None, None
    
    try:
        with open(json_file_path, 'r') as f:
            metadata = json.load(f)
        
        pulse_sequence = metadata.get('PulseSequenceName', '')

        with open("pulse_sequence_examples.txt", 'a') as f:
            f.write(json.dumps(metadata) + "\n")

        # Map pulse sequence names to BIDS contrasts and acquisition info
        pulse_sequence_mappings = {
            'STIR': ('T2w', 'stir'),
            'FLAIR': ('T1w', 'flair'),
            'FSE': ('T1w', 'fse'),
            'TSE': ('T1w', 'tse'),
            'FRFSE': ('T2w', 'frfse'),
            'RST': ('T2w', 'rst'),
            'T1': ('T1w', ''),
            'T2': ('T2w', ''),
            'TIRM': ('T2w', 'tirm'),  # Another name for STIR
            'SPGR': ('T1w', 'spgr'),  # Spoiled Gradient Echo
            'GRE': ('T1w', 'gre'),    # Gradient Echo
        }
        
        for sequence_name, (contrast, acq_info) in pulse_sequence_mappings.items():
            if sequence_name in pulse_sequence.upper():
                logger.info(f"Found contrast from JSON: {pulse_sequence} -> {contrast} (acq: {acq_info})")
                return contrast, acq_info
        
        logger.warning(f"Unknown pulse sequence in JSON: {pulse_sequence}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file_path}: {str(e)}")
        return None, None


def normalize_filename_with_json(filename, json_contrast_info=None):
    """
    Normalize filename to follow BIDS conventions with camelCase acquisition tags and separate chunk keys
    :param filename: original filename
    :param json_contrast_info: tuple of (contrast_suffix, acquisition_info) from JSON sidecar
    :return: normalized BIDS filename
    """
    # Define chunk mappings for numbered variants
    chunk_mappings = {
        '1': 'T12toL4',
        '2': 'L4toSI'
    }
    
    # Handle numbered variants and map to chunks
    # Check for T1w-X or T2w-X patterns
    t1w_numbered = re.search(r'_T1w-(\d+)', filename)
    t2w_numbered = re.search(r'_T2w-(\d+)', filename)
    frfse_numbered = re.search(r'_T2w_FRFSE-(\d+)', filename)
    stir_numbered = re.search(r'_STIR-(\d+)', filename)
    
    chunk_info = ""
    
    if t1w_numbered:
        number = t1w_numbered.group(1)
        if number in chunk_mappings:
            chunk_info = f"_chunk-{chunk_mappings[number]}"
        filename = re.sub(r'_T1w-\d+', '_T1w', filename)
    elif t2w_numbered:
        number = t2w_numbered.group(1)
        if number in chunk_mappings:
            chunk_info = f"_chunk-{chunk_mappings[number]}"
        filename = re.sub(r'_T2w-\d+', '_T2w', filename)
    elif frfse_numbered:
        number = frfse_numbered.group(1)
        if number in chunk_mappings:
            chunk_info = f"_chunk-{chunk_mappings[number]}"
        filename = re.sub(r'_T2w_FRFSE-\d+', '_T2w_FRFSE', filename)
    elif stir_numbered:
        number = stir_numbered.group(1)
        if number in chunk_mappings:
            chunk_info = f"_chunk-{chunk_mappings[number]}"
        filename = re.sub(r'_STIR-\d+', '_STIR', filename)
    
    # Handle _discs-X patterns and convert to _desc-discX
    discs_match = re.search(r'_discs-(\d+)', filename)
    desc_info = ""
    if discs_match:
        disc_number = discs_match.group(1)
        desc_info = f"_desc-disc{disc_number}"
        filename = re.sub(r'_discs-\d+', '', filename)
    
    # Define contrast mappings to T1w or T2w
    contrast_mappings = {
        'STIR': 'T2w',
        'T1w_FLAIR': 'T1w',
        'T1w_FSE': 'T1w', 
        'T1w_TSE': 'T1w',
        'T2w_FRFSE': 'T2w',
        'T2w_RST': 'T2w',
        'T2w_FRFSE_discs': 'T2w',
        'T1w': 'T1w',
        'T2w': 'T2w'
    }
    
    # Extract acquisition tag and contrast
    acq_pattern = r'_acq-([^_]+)_(.+?)(?:\.(nii|json))?$'
    match = re.search(acq_pattern, filename)
    
    if match:
        acq_part = match.group(1)
        contrast_part = match.group(2)
        extension = match.group(3)
        
        # Handle chunk extraction from acquisition tag (in addition to numbered variants)
        acq_chunk_info = chunk_info  # Use chunk from numbered variants if present
        acq_base = acq_part
        
        # Check if acquisition contains chunk information
        if 'chunk' in acq_part and not acq_chunk_info:
            # Pattern: axial-chunk-L4toS1 -> acq_base=axial, chunk=L4toS1
            chunk_match = re.search(r'(.+?)-chunk-(.+)', acq_part)
            if chunk_match:
                acq_base = chunk_match.group(1)
                acq_chunk_info = f"_chunk-{chunk_match.group(2)}"
        
        # Convert acquisition base to camelCase (without chunk part)
        acq_parts = acq_base.split('-')
        if len(acq_parts) > 1:
            acq_camel = acq_parts[0] + ''.join(word.capitalize() for word in acq_parts[1:])
        else:
            acq_camel = acq_parts[0]
        
        # Determine the BIDS contrast suffix and update acquisition tag
        bids_contrast = None
        updated_acq = acq_camel
        
        # First try to find contrast from filename
        for original_contrast, bids_suffix in contrast_mappings.items():
            if contrast_part == original_contrast:
                bids_contrast = bids_suffix
                # Add contrast info to acquisition tag if it's not T1w or T2w
                if original_contrast not in ['T1w', 'T2w']:
                    # Convert contrast to camelCase and add to acquisition
                    contrast_camel = original_contrast.replace('_', '').replace('T1w', '').replace('T2w', '')
                    if contrast_camel:
                        # Convert sequences like 'FLAIR', 'FSE', 'FRFSE', 'STIR' to camelCase
                        contrast_camel = contrast_camel.lower().capitalize()
                        updated_acq = acq_camel + contrast_camel
                break
        
        # If no contrast found in filename, use JSON information
        if not bids_contrast and json_contrast_info:
            json_contrast, json_acq_info = json_contrast_info
            bids_contrast = json_contrast
            if json_acq_info:
                updated_acq = acq_camel + json_acq_info.capitalize()
            logger.info(f"Using contrast from JSON: {bids_contrast} for file: {filename}")
        
        if bids_contrast:
            # Reconstruct filename with chunk and desc keys if present
            base_filename = re.sub(acq_pattern, '', filename)
            
            # Handle gzipping for NIfTI files
            if extension == 'nii':
                extension = 'nii.gz'
            
            if extension:
                new_filename = f"{base_filename}_acq-{updated_acq}{acq_chunk_info}{desc_info}_{bids_contrast}.{extension}"
            else:
                new_filename = f"{base_filename}_acq-{updated_acq}{acq_chunk_info}{desc_info}_{bids_contrast}"
            return new_filename
    
    # Fallback: if no acquisition tag found, try to handle basic contrast replacement
    for original_contrast, bids_suffix in contrast_mappings.items():
        if f'_{original_contrast}' in filename:
            if original_contrast not in ['T1w', 'T2w']:
                # Add acquisition tag for non-standard contrasts
                contrast_camel = original_contrast.replace('_', '').replace('T1w', '').replace('T2w', '').lower().capitalize()
                replacement = f'_acq-{contrast_camel.lower()}{chunk_info}{desc_info}_{bids_suffix}'
            else:
                replacement = f'{chunk_info}{desc_info}_{bids_suffix}'
            
            # Handle gzipping for NIfTI files
            if filename.endswith('.nii'):
                filename = filename.replace('.nii', '.nii.gz')
            
            filename = filename.replace(f'_{original_contrast}', replacement)
            break
    else:
        # If no contrast found in filename, use JSON information as fallback
        if json_contrast_info:
            json_contrast, json_acq_info = json_contrast_info
            
            # Extract base filename without extension
            base_filename = filename
            extension = ''
            if filename.endswith('.nii'):
                base_filename = filename[:-4]
                extension = 'nii.gz'
            elif filename.endswith('.json'):
                base_filename = filename[:-5]
                extension = 'json'
            
            # Add acquisition info based on JSON
            if json_acq_info:
                acq_tag = f"_acq-{json_acq_info}"
            else:
                acq_tag = ""
            
            # Reconstruct filename
            if extension:
                filename = f"{base_filename}{acq_tag}{chunk_info}{desc_info}_{json_contrast}.{extension}"
            else:
                filename = f"{base_filename}{acq_tag}{chunk_info}{desc_info}_{json_contrast}"
            
            logger.info(f"Applied contrast from JSON to filename: {filename}")
    
    return filename


def process_subject_session(subject_folder, session_folder, path_output, sub_dict_tsv, error_files):
    """
    Process all files for a specific subject and session
    :param subject_folder: subject folder name (e.g., sub-nMRI001)
    :param session_folder: session folder name (e.g., ses-Pre)
    :param path_output: output BIDS folder path
    :param sub_dict_tsv: dictionary to track subjects for participants.tsv
    :param error_files: list to track files that couldn't be processed
    """
    subject_session_path = os.path.join(path_dataset, subject_folder, session_folder)
    
    if not os.path.isdir(subject_session_path):
        return
        
    # Add subject to participants dictionary
    source_id = subject_folder.replace('sub-', '')
    if subject_folder not in sub_dict_tsv:
        sub_dict_tsv[subject_folder] = source_id
    
    # Process all files in the session folder
    for filename in os.listdir(subject_session_path):
        file_path = os.path.join(subject_session_path, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        # Skip Excel files and CSV files
        if filename.endswith('.xlsx') or filename.endswith('.csv'):
            continue
            
        try:
            # Get contrast information from JSON sidecar if available
            json_contrast_info = None
            if filename.endswith('.nii'):
                # Look for corresponding JSON file
                json_filename = filename.replace('.nii', '.json')
                json_file_path = os.path.join(subject_session_path, json_filename)
                json_contrast_info = get_contrast_from_json(json_file_path)
            elif filename.endswith('.json'):
                # Get contrast info from the JSON file itself
                json_contrast_info = get_contrast_from_json(file_path)
            
            # Normalize filename for BIDS compliance with JSON contrast info
            normalized_filename = normalize_filename_with_json(filename, json_contrast_info)
            
            # Determine output folder based on file type
            if any(contrast in normalized_filename for contrast in ['_T1w', '_T2w']):
                output_subfolder = 'anat'
            else:
                # Skip files that don't match expected contrasts
                logger.warning(f'Skipping file with unrecognized contrast: {filename}')
                error_files.append(f"{subject_folder}/{session_folder}/{filename} - Unrecognized contrast")
                continue
            
            # Create output directory structure
            path_subject_session_out = os.path.join(path_output, subject_folder, session_folder, output_subfolder)
            create_subject_folder_if_not_exists(path_subject_session_out)
            
            # Output file path
            path_file_out = os.path.join(path_subject_session_out, normalized_filename)
            
            # Process based on file type
            if filename.endswith('.nii'):
                # Load image, reorient to RPI, and save
                try:
                    img = Image(file_path).change_orientation('RPI')
                    img.save(path_file_out)
                    logger.info(f'Processed and saved: {file_path} --> {path_file_out}')
                except Exception as e:
                    logger.error(f'Error processing image {file_path}: {str(e)}')
                    error_files.append(f"{subject_folder}/{session_folder}/{filename} - Image processing error: {str(e)}")
                    
            elif filename.endswith('.json'):
                # Copy JSON sidecar files
                try:
                    copy_file(file_path, path_file_out)
                except Exception as e:
                    logger.error(f'Error copying JSON file {file_path}: {str(e)}')
                    error_files.append(f"{subject_folder}/{session_folder}/{filename} - JSON copy error: {str(e)}")
            else:
                logger.warning(f'Skipping file with unrecognized extension: {filename}')
                error_files.append(f"{subject_folder}/{session_folder}/{filename} - Unrecognized file extension")
                
        except Exception as e:
            logger.error(f'Error processing file {file_path}: {str(e)}')
            error_files.append(f"{subject_folder}/{session_folder}/{filename} - General processing error: {str(e)}")


def write_error_log(error_files, path_output):
    """
    Write error log file with list of files that couldn't be processed
    :param error_files: list of error messages for files that couldn't be processed
    :param path_output: path to the output BIDS folder
    """
    error_file_path = os.path.join(path_output, 'err.txt')
    
    with open(error_file_path, 'w') as f:
        f.write(f"BIDS Conversion Error Log - {datetime.datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
        
        if error_files:
            f.write(f"Total files with errors: {len(error_files)}\n\n")
            for error in error_files:
                f.write(f"{error}\n")
        else:
            f.write("No errors encountered during conversion.\n")
    
    logger.info(f'Error log written to {error_file_path}')
    if error_files:
        logger.warning(f'{len(error_files)} files could not be processed - see {error_file_path} for details')


def main():
    global path_dataset
    
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
    
    fh = logging.FileHandler(FNAME_LOG)
    logging.root.addHandler(fh)
    logger.info("INFO: log file will be saved to {}".format(FNAME_LOG))

    # Print current time and date to log file
    logger.info('\nAnalysis started at {}'.format(datetime.datetime.now()))
    
    # Read demographics file
    demographics_file = os.path.join(path_dataset, 'demographics.csv')
    demographics = read_demographics(demographics_file)
    
    # Initialize dict for participants.tsv and error tracking
    sub_dict_tsv = dict()
    error_files = []
    
    # Process each subject folder
    for item in os.listdir(path_dataset):
        item_path = os.path.join(path_dataset, item)
        
        # Skip non-directories and non-subject folders
        if not os.path.isdir(item_path) or not item.startswith('sub-'):
            continue
            
        subject_folder = item
        logger.info(f'Processing subject: {subject_folder}')
        
        # Process each session for this subject
        for session_item in os.listdir(item_path):
            session_path = os.path.join(item_path, session_item)
            
            # Skip non-directories and non-session folders
            if not os.path.isdir(session_path) or not session_item.startswith('ses-'):
                continue
                
            session_folder = session_item
            logger.info(f'Processing session: {session_folder}')
            
            # Process all files in this subject/session
            process_subject_session(subject_folder, session_folder, path_output, sub_dict_tsv, error_files)
    
    # Create BIDS metadata files
    participants_tsv_list = list(sub_dict_tsv.items())
    create_participants_tsv(participants_tsv_list, demographics, path_output)
    create_participants_json(path_output)
    create_dataset_description(path_output)
    copy_script(path_output)
    
    # Write error log
    write_error_log(error_files, path_output)
    
    logger.info('\nConversion completed successfully!')


if __name__ == "__main__":
    main()
