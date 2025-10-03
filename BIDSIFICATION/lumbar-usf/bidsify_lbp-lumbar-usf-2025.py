import os
import sys
import shutil
import json
import argparse
import logging
import datetime
import csv
import re
import pandas as pd
from vrac.data_management.image import Image

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

def get_parser():
    parser = argparse.ArgumentParser(description='Convert lbp-lumbar-usf-2025 dataset to BIDS format.')
    parser.add_argument("-i", "--path-dataset",
                        help="Path to the sourcedata folder containing the non-BIDS dataset",
                        required=True)
    parser.add_argument("-o", "--path-output",
                        help="Path to the output folder where the BIDS dataset will be stored.",
                        required=True)
    return parser

def create_folder(path):
    """Create folder if it doesn't exist"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f'Creating directory: {path}')

def write_json(path_output, json_filename, data_json):
    """Write JSON file with proper formatting"""
    with open(os.path.join(path_output, json_filename), 'w') as json_file:
        json.dump(data_json, json_file, indent=4)
        json_file.write("\n")
        logger.info(f'{json_filename} created in {path_output}')

def normalize_filename(filename, subject, session):
    """
    Normalize filename according to BIDS conventions from README
    Ensures proper entity order and formatting
    """
    # Start with the original filename
    normalized = filename
    
    # Handle sequence type suffixes - extract and move to acq field
    sequence_patterns = {
        '_T1w_FSE': ('_T1w', 'fse'),
        '_T1_FSE': ('_T1w', 'fse'),
        '_T1w_TSE': ('_T1w', 'tse'), 
        '_T2w_TSE': ('_T2w', 'tse'), 
        '_T1w_RST': ('_T1w', 'rst'), 
        '_T2w_obl': ('_T2w', 'obl'), 
        '_T2w_FRFSE': ('_T2w', 'frfse'),
        '_T2w_STIR.': ('_T2w.', 'stir'),
        '_T2w_STIR-': ('_T2w-', 'stir'),
        '_STIR_irFSE.': ('_T2w.', 'stirIrFse'),
        '_T2w_3D': ('_T2w', '3d'),
        '_T2w_3DSpace': ('_T2w', '3dSpace'),
        '_T2w_SPACE': ('_T2w', 'space'),
        '_T2w_FS.': ('_T2w.', 'fs'),
        '_STIR.': ('_T2w.', 'stir'),
        '_FLAIR.': ('_T1w.', 'flair'),
        '_STIR_IRFSE': ('_T2w', 'stirirfse'),
        '_T2w_FSE.': ('_T2w.', 'fse'),
        '_T2w_FSE-': ('_T2w-', 'fse'),
        '_T2w_RST': ('_T2w', 'rst'),
        '_T2w_WTD': ('_T2w', 'wtd'),
        '_T1w_WTD': ('_T1w', 'wtd'),
        '_T2w_STIRFSE': ('_T2w', 'stirfse'),
        '_T2w_ANGLED': ('_T2w', 'angled'),
        '_T2w-angled': ('_T2w', 'angled'),
        '_T2w_angled': ('_T2w', 'angled'),
        '_acq-axial_3D': ('_T2w', 'axial3d'),
        '_acq-sag_3D': ('_T2w', 'sagittal3d'),
        '_T2w_DNE': ('_T2w', 'dne'),
        '_T1w_DNE': ('_T1w', 'dne'),
        '_STIR_DNE': ('_T2w', 'stirdne'),
        '_SAGITTAL_T2_FS': ('_T2w', 'FSsagittal'),
        '_SAGITTAL_T1': ('_T1w', 'sagittal'),
        '_AXIAL_T2': ('_T2w', 'axial'),
        '_SAGITTAL_STIR': ('_T2w', 'sagittalStir'),
        '_fat.': ('_T2w.', 'dixonFat'),
        '_fatfrac.': ('_T2w.', 'dixonFatFrac'),
        '_inphase.': ('_T2w.', 'dixonInphase'),
        '_outphase.': ('_T2w.', 'dixonOutphase'),
        '_acq-axial_water': ('_T2w', 'dixonWater'),
        '_FSE_IR.': ('_T2w.', 'stirFse'),
        '_T2w_tirm.': ('_T2w.', 'tirm'),
        '_T2w_tse.': ('_T2w.', 'tse'),
        '_acq-sagittal-T2w': ('_T2w', 'sagittal'),
        '_acq-axial-T2w': ('_T2w', 'axial'),
    }
    
    for pattern, (contrast, seq_type) in sequence_patterns.items():
        if pattern in normalized:
            normalized = normalized.replace(pattern, contrast)
            # Add to existing acq or create new acq
            if '_acq-' in normalized:
                normalized = re.sub(r'_acq-([^_]+)', f'_acq-\\1{seq_type.capitalize()}', normalized)
            else:
                normalized = normalized.replace(f'{contrast}', f'_acq-{seq_type}_{contrast[1:]}')

    # Handle disc information: _discs-X -> _desc-discX
    normalized = re.sub(r'_discs-(\d+)', r'_desc-disc\1', normalized)
    
    # Handle numbered variants for chunks (_T1w-1, _T2w-1, ..., _T1w-9, _T2w-9 -> _chunk-X)
    for i in range(1, 15):
        pattern = fr'_T([12])w-{i}[.]'
        replacement = fr'_chunk-{i}_T\1w.'
        if re.search(pattern, normalized):
            normalized = re.sub(pattern, replacement, normalized)
            break  # Only apply one chunk replacement per filename

    # Special case for nMRI_034_Pre_AXIAL_T2
    if '_e1a.' in normalized:
        normalized = normalized.replace('_e1a.', '_chunk-2.')
    elif '_e1.' in normalized:
        normalized = normalized.replace('_e1.', '_chunk-1.')
    
    # Fix chunk formatting: ensure _chunk-XXX_ not _chunk-XXX-YYY_ or -chunk-
    # Convert _acq-YYY-chunk-XXX_ to _acq-YYY_chunk-XXX_
    normalized = re.sub(r'_acq-([^_]+)-chunk-([^_]+)', r'_acq-\1_chunk-\2', normalized)
    
    # Extract the file extension first
    ext_match = re.search(r'\.(nii(?:\.gz)?|json)$', normalized)
    if ext_match:
        extension = ext_match.group(1)
        # Remove extension for processing
        base_filename = normalized[:ext_match.start()]
    else:
        raise ValueError(f"Filename {filename} does not have a valid .nii, .nii.gz, or .json extension.")
    
    # Extract all entities from the base filename
    entities = {}
    
    entities['sub'] = subject
    entities['ses'] = session

    # Extract other entities
    acq_match = re.search(r'_acq-([^_]+)', base_filename)
    chunk_match = re.search(r'_chunk-([^_]+)', base_filename)
    desc_match = re.search(r'_desc-([^_]+)', base_filename)
    run_match = re.search(r'_run-([^_]+)', base_filename)
    
    # Extract contrast (T1w, T2w) - can appear anywhere in the base filename
    contrast_match = re.search(r'_(T[12]w)(?:_|$)', base_filename)
    
    # Build filename in proper BIDS order
    filename_parts = []
    
    if entities.get('sub'):
        filename_parts.append(entities['sub'])
    if entities.get('ses'):
        filename_parts.append(entities['ses'])
    if acq_match:
        filename_parts.append(f"acq-{acq_match.group(1)}")
    if chunk_match:
        filename_parts.append(f"chunk-{chunk_match.group(1)}")
    if desc_match:
        filename_parts.append(f"desc-{desc_match.group(1)}")
    if run_match:
        filename_parts.append(f"run-{run_match.group(1)}")
    
    # Add contrast last (before extension)
    if contrast_match:
        filename_parts.append(contrast_match.group(1))
    
    # Join with underscores and add extension
    if filename_parts:
        normalized = '_'.join(filename_parts)
    else:
        # Fallback to original base filename if no entities found
        normalized = base_filename
    
    # Add the extension
    normalized += '.' + extension
    
    # Change .nii to .nii.gz for images
    if normalized.endswith('.nii'):
        normalized = normalized.replace('.nii', '.nii.gz')

    if not normalized.startswith('sub-'):
        raise ValueError(f"Normalized filename {normalized} does not start with 'sub-'. Original filename: {filename}")
    
    return normalized

def load_demographics(path_dataset):
    """Load demographics from CSV file"""
    demographics = {}
    csv_path = os.path.join(path_dataset, 'demographics.csv')
    excel_path = os.path.join(path_dataset, 'USF nMRI Demographics.xlsx')
    
    # Try CSV first, then Excel as fallback
    demo_file = csv_path if os.path.exists(csv_path) else excel_path if os.path.exists(excel_path) else None
    
    if demo_file:
        try:
            if demo_file.endswith('.csv'):
                df = pd.read_csv(demo_file)
                logger.info(f"Loaded demographics from {demo_file}")
            else:
                df = pd.read_excel(demo_file)
                logger.info(f"Loaded demographics from {demo_file}")
            
            # Process each row to extract demographics
            for _, row in df.iterrows():
                # Extract subject ID from 'Subject ID' column
                subject_num = str(row.get('Subject ID', ''))
                if not subject_num or subject_num == 'nan':
                    continue
                
                # Format as BIDS subject ID (pad with zeros to match nMRI format)
                subject_id = f"sub-nMRI{subject_num.zfill(3)}"
                
                # Extract other demographics
                age = row.get('Age', 'n/a')
                sex = row.get('Sex', 'n/a')
                level_dx = row.get('Level Dx', 'n/a')
                dx_code = row.get('Dx Code\n(ICD-10 / ICD-9)', 'n/a')
                level_treated = row.get('Level Treated\n(e.g. L4, L5)', 'n/a')
                
                # Handle NaN values
                age = str(age) if pd.notna(age) else 'n/a'
                sex = str(sex) if pd.notna(sex) else 'n/a'
                level_dx = str(level_dx) if pd.notna(level_dx) else 'n/a'
                dx_code = str(dx_code) if pd.notna(dx_code) else 'n/a'
                level_treated = str(level_treated) if pd.notna(level_treated) else 'n/a'
                
                # Create notes from diagnosis and treatment info
                notes_parts = []
                if dx_code != 'n/a':
                    notes_parts.append(f"Dx Code: {dx_code}")
                if level_treated != 'n/a':
                    notes_parts.append(f"Level Treated: {level_treated}")
                
                notes = "; ".join(notes_parts) if notes_parts else 'n/a'
                
                # Store demographics (only store unique subject info, not per session)
                if subject_id not in demographics:
                    demographics[subject_id] = {
                        'age': age,
                        'sex': sex,
                        'pathology': 'LBP',  # Low Back Pain for all subjects
                        'institution': 'USF',
                        'level_dx': level_dx,
                        'notes': notes
                    }
            
        except Exception as e:
            logger.warning(f"Could not load demographics: {e}")
    else:
        logger.warning("No demographics file found (demographics.csv or USF nMRI Demographics.xlsx)")
    
    return demographics

def create_participants_tsv(participants_list, demographics, path_output):
    """Create participants.tsv file with demographics"""
    with open(os.path.join(path_output, 'participants.tsv'), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['participant_id', 'source_id', 'species', 'age', 'sex', 'pathology', 'institution', 'level_dx', 'notes'])
        
        species = 'homo sapiens'
        
        for participant_id, source_id in sorted(participants_list, key=lambda x: x[0]):
            demo = demographics.get(participant_id, {})
            
            row = [
                participant_id,
                source_id, 
                species,
                demo.get('age', 'n/a'),
                demo.get('sex', 'n/a'),
                demo.get('pathology', 'LBP'),
                demo.get('institution', 'USF'),
                demo.get('level_dx', 'n/a'),
                demo.get('notes', 'n/a')
            ]
            tsv_writer.writerow(row)
        
        logger.info(f'participants.tsv created in {path_output}')

def create_participants_json(path_output):
    """Create participants.json file"""
    data_json = {
        "participant_id": {"Description": "Unique Participant ID", "LongName": "Participant ID"},
        "source_id": {"Description": "Original subject name"},
        "species": {"Description": "Binomial species name of participant", "LongName": "Species"},
        "age": {"Description": "Participant age", "LongName": "Participant age", "Units": "years"},
        "sex": {"Description": "sex of the participant as reported by the participant", 
                "Levels": {"M": "male", "F": "female", "O": "other"}},
        "pathology": {"Description": "The diagnosis of pathology of the participant", 
                     "LongName": "Pathology name", 
                     "Levels": {
                         "HC": "Healthy Control",
                         "LBP": "Low Back Pain",
                         "NRC": "Nerve Root Compression", 
                         "DCM": "Degenerative Cervical Myelopathy",
                         "MS": "Multiple Sclerosis",
                         "SCI": "Traumatic Spinal Cord Injury"
                     }},
        "institution": {"Description": "Human-friendly institution name", "LongName": "BIDS Institution ID"},
        "level_dx": {"Description": "Level of diagnosis for the participant", "LongName": "Level of Diagnosis"},
        "notes": {"Description": "Additional notes about the participant", "LongName": "Additional notes"}
    }
    write_json(path_output, 'participants.json', data_json)

def create_dataset_description(path_output):
    """Create dataset_description.json file"""
    data_json = {
        "BIDSVersion": "1.9.0",
        "Name": "lbp-lumbar-usf-2025",
        "DatasetType": "raw"
    }
    write_json(path_output, 'dataset_description.json', data_json)

def copy_script(path_output):
    """Copy the script itself to the path_output/code folder"""
    path_script_in = sys.argv[0]
    path_code = os.path.join(path_output, 'code')
    create_folder(path_code)
    path_script_out = os.path.join(path_code, os.path.basename(path_script_in))
    logger.info(f'Copying {path_script_in} to {path_script_out}')
    shutil.copyfile(path_script_in, path_script_out)

def main(path_dataset, path_output):

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
    
    # Load demographics
    demographics = load_demographics(path_dataset)
    
    # Initialize tracking variables
    participants_list = []
    error_files = []
    processed_images = 0
    processed_jsons = 0
    
    # Walk through all files in sourcedata
    for root, dirs, files in os.walk(path_dataset):
        # Skip root directory files like Excel
        if root == path_dataset:
            continue
            
        for fname in files:
            # Skip non-image and non-json files
            if not (fname.endswith('.nii') or fname.endswith('.json')):
                continue
                
            path_file_in = os.path.join(root, fname)
            
            # Parse directory structure to get subject and session
            rel_path = os.path.relpath(root, path_dataset)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) < 2:
                raise ValueError(f"SKIPPED - Unexpected path structure: {path_file_in}")
            
            subject = path_parts[0]  # e.g., sub-nMRI001
            session = path_parts[1]  # e.g., ses-Pre, ses-Post
            
            # Add to participants list
            if (subject, subject) not in participants_list:
                participants_list.append((subject, subject))
            
            # Create output directory structure
            anat_dir = os.path.join(path_output, subject, session, 'anat')
            create_folder(anat_dir)
                
            try:
                # Normalize filename
                normalized_fname = normalize_filename(fname, subject, session)
                path_file_out = os.path.join(anat_dir, normalized_fname)

                if '_T1w' in normalized_fname or '_T2w' in normalized_fname:
                    # Check for overwrite and add run number if needed
                    if os.path.exists(path_file_out):
                        # Extract base filename without extension
                        base_name, ext = os.path.splitext(normalized_fname)
                        if ext == '.gz':
                            base_name, ext2 = os.path.splitext(base_name)
                            ext = ext2 + ext
                        
                        # Find next available run number
                        run_num = 1
                        while True:
                            # Insert run entity before contrast suffix
                            if '_T1w' in base_name:
                                new_base = base_name.replace('_T1w', f'_run-{run_num:02d}_T1w')
                            elif '_T2w' in base_name:
                                new_base = base_name.replace('_T2w', f'_run-{run_num:02d}_T2w')
                            else:
                                new_base = f"{base_name}_run-{run_num:02d}"
                            
                            new_path = os.path.join(anat_dir, new_base + ext)
                            if not os.path.exists(new_path):
                                path_file_out = new_path
                                normalized_fname = new_base + ext
                                logger.info(f'File exists, using run number: {normalized_fname}')
                                break
                            run_num += 1
                            
                            # Safety check to prevent infinite loop
                            if run_num > 99:
                                raise ValueError(f"ERROR - Too many runs (>99) for {path_file_in}")
                    
                    if fname.endswith('.nii'):
                        # Process NIfTI image: load, reorient to RPI, and save as .nii.gz
                        logger.info(f'Processing image: {path_file_in}')
                        img = Image(path_file_in).change_orientation('RPI')
                        img.save(path_file_out)
                        logger.info(f'Saved: {path_file_out}')
                        processed_images += 1
                        
                    elif fname.endswith('.json'):
                        # Copy JSON sidecar file
                        logger.info(f'Copying JSON: {path_file_in} -> {path_file_out}')
                        shutil.copy(path_file_in, path_file_out)
                        processed_jsons += 1
                else:
                    raise ValueError("Unknown contrast")
                    
            except Exception as e:
                error_msg = f"ERROR - Failed to process {path_file_in}: {str(e)}"
                logger.error(error_msg)
                
                # Copy error files to sourcedata folder with bids structure
                try:
                    # Create sourcedata directory in output
                    sourcedata_dir = os.path.join(path_output, 'sourcedata')
                    source_anat_dir = os.path.join(sourcedata_dir, subject, session, 'anat')

                    create_folder(source_anat_dir)

                    source_file_out = os.path.join(source_anat_dir, fname)

                    if fname.endswith('.nii'):
                        # Process NIfTI image: load, reorient to RPI, and save as .nii.gz
                        logger.info(f'Processing image: {path_file_in}')
                        img = Image(path_file_in).change_orientation('RPI')
                        img.save(source_file_out.replace('.nii', '.nii.gz'))
                        logger.info(f'Saved: {source_file_out}')
                        processed_images += 1
                        
                    elif fname.endswith('.json'):
                        # Copy JSON sidecar file
                        logger.info(f'Copying JSON: {path_file_in} -> {source_file_out}')
                        shutil.copy(path_file_in, source_file_out)
                        processed_jsons += 1

                except Exception as copy_error:
                    logger.warning(f'Failed to copy error file to sourcedata: {copy_error}')
                    error_files.append(error_msg)
    
    # Create BIDS metadata files
    create_participants_tsv(participants_list, demographics, path_output)
    create_participants_json(path_output)
    create_dataset_description(path_output)
    copy_script(path_output)
    
    logger.info(f'\nBIDS conversion completed at {datetime.datetime.now()}')
    logger.info(f'Processed {len(participants_list)} participants')
    logger.info(f'Successfully processed {processed_images} NIfTI images')
    logger.info(f'Successfully processed {processed_jsons} JSON files')
    if error_files:
        logger.warning(f'{len(error_files)} files had errors - check err.txt')
    else:
        logger.info('All files processed successfully!')

if __name__ == "__main__":
    # parser = get_parser()
    # args = parser.parse_args()
    
    path_dataset = "/Users/nathan/data/lumbar-usf-mix/sourcedata" # os.path.abspath(args.path_dataset)
    path_output = "/Users/nathan/data/lumbar-usf-mix/lbp-lumbar-usf-2025" #os.path.abspath(args.path_output)
    main(path_dataset, path_output)