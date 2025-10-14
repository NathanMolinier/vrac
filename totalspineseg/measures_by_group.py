import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import copy
from pathlib import Path

def main():
    folder_path = Path('/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/metrics_output')
    demographics_path = Path('/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/lbp-lumbar-usf-2025/participants.tsv')

    demographics = pd.read_csv(demographics_path, sep='\t')

    all_values = {}
    all_demographics = {}
    for sub in os.listdir(folder_path):
        sub_folder = folder_path / sub
        sub_name = sub.split('_')[0]
        sub_info = demographics[demographics['participant_id'] == sub_name]
        sub_info = df_to_dict(sub_info)

        # Compute metrics subject
        control_data = compute_metrics_subject(sub_folder)

        # Gather all values for each metric and structures
        for struc in control_data.keys():
            for struc_name in control_data[struc].keys():
                for metric in control_data[struc][struc_name].keys():
                    # Add subject to all_values
                    subject_value = control_data[struc][struc_name][metric]
                    if subject_value != -1 and isinstance(sub_info['age'], (int, float)) and sub_info['sex'].strip() in ['M', 'F']:
                        if struc not in all_values:
                            all_values[struc] = {}
                            all_demographics[struc] = {}
                        if struc_name not in all_values[struc]:
                            all_values[struc][struc_name] = {}
                            all_demographics[struc][struc_name] = {}
                        if metric not in all_values[struc][struc_name]:
                            all_values[struc][struc_name][metric] = []
                            all_demographics[struc][struc_name][metric] = []
                        all_values[struc][struc_name][metric].append(subject_value)
                        all_demographics[struc][struc_name][metric].append(sub_info)
        
    # Align canal and CSF for control group
    all_values = rescale_canal(all_values)

    print()

def df_to_dict(df):
    idx = df['participant_id'].keys()[0]
    d = {k:v[idx] for k,v in df.to_dict().items()}
    return d

def compute_metrics_subject(subject_folder):
    """
    Compute metrics for a single subject and return merged_data dict for global figures.

    Parameters:
        subject_folder (Path): Path to the subject's metrics folder.
        ofolder_path (Path): Path to the output folder where reports will be saved.
        quiet (bool, optional): If True, suppresses output messages. Defaults to False.

    Returns:
        dict: A dictionary containing merged metrics data for the subject.
    """
    merged_data = {}

    # List of expected CSV files
    csv_files = {
        "canal":process_canal, 
        "csf":process_csf, 
        "discs":process_discs, 
        "foramens":process_foramens, 
        "vertebrae":process_vertebrae
    }

    # Load each CSV if it exists
    for csv_file, process_func in csv_files.items():
        csv_path = subject_folder / 'csv' / f"{csv_file}.csv"
        if csv_path.exists():
            subject_data = pd.read_csv(str(csv_path))
            # Call the compute function to process the data
            merged_data[csv_file] = process_func(subject_data)
    
    # Compute discs metrics
    merged_data = compute_discs_metrics(merged_data)

    # Compute foramen metrics
    merged_data = compute_foramens_metrics(merged_data)

    # Compute vertebrae metrics
    merged_data = compute_vertebrae_metrics(merged_data)
    return merged_data

def process_canal(subject_data):
    # Convert pandas columns to lists
    canal_dict = {'canal': {}, 'spinalcord': {}, 'spinalcord/canal': {}}
    for column in subject_data.columns[2:]:
        if column not in ['canal_centroid', 'angle_AP', 'angle_RL', 'length']:
            if not 'canal' in column:
                canal_dict['spinalcord'][column.replace('_spinalcord','')] = subject_data[column].tolist()
            if not 'spinalcord' in column:
                canal_dict['canal'][column.replace('_canal','')] = subject_data[column].tolist()
    
    # Create spinalcord/canal quotient
    for key in canal_dict['spinalcord'].keys():
        if not key in ['slice_nb', 'disc_level']:
            canal_dict['spinalcord/canal'][key] = []
            for i in range(len(canal_dict['spinalcord'][key])):
                canal_value = canal_dict['canal'][key][i]
                spinalcord_value = canal_dict['spinalcord'][key][i]
                if canal_value != 0 and canal_value != -1 and spinalcord_value != -1:
                    canal_dict['spinalcord/canal'][key].append(spinalcord_value / canal_value)
                else:
                    canal_dict['spinalcord/canal'][key].append(-1)
        else:
            canal_dict['spinalcord/canal'][key] = canal_dict['spinalcord'][key]
    return canal_dict

def process_csf(subject_data):
    # Convert pandas columns to lists
    csf_dict = {'csf': {}}
    for column in subject_data.columns[2:]:
        csf_dict['csf'][column] = subject_data[column].tolist()
    return csf_dict

def process_discs(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def process_vertebrae(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data, intensity_profile=False)
    return subject_dict

def process_foramens(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_discs_metrics(data_dict):
    # Compute Disc Height Index (DHI)
    for struc_name in data_dict['discs'].keys():
        top_vertebra = struc_name.split('-')[0]
        if top_vertebra in data_dict['vertebrae']:
            # Normalize disc height with top vertebra AP_thickness
            data_dict['discs'][struc_name]['DHI'] = data_dict['discs'][struc_name]['median_thickness'] / data_dict['vertebrae'][top_vertebra]['AP_thickness']

            # Normalize disc volume with top vertebra volume
            data_dict['discs'][struc_name]['volume'] = data_dict['discs'][struc_name]['volume'] / data_dict['vertebrae'][top_vertebra]['volume']
        else:
            data_dict['discs'][struc_name]['DHI'] = -1
            data_dict['discs'][struc_name]['volume'] = -1
    return data_dict

def compute_foramens_metrics(data_dict):
    # Compute Foramen metrics
    for struc_name in data_dict['foramens'].keys():
        top_vertebra = struc_name.replace('foramens_','').split('-')[0]
        if not top_vertebra in data_dict['vertebrae']:
            data_dict['foramens'][struc_name]['right_surface'] = -1
            data_dict['foramens'][struc_name]['left_surface'] = -1
            data_dict['foramens'][struc_name]['asymmetry R/L'] = -1
        else:
            # Normalize foramen surfaces with top vertebra AP thickness
            for surface in ['right_surface', 'left_surface']:
                data_dict['foramens'][struc_name][surface] = data_dict['foramens'][struc_name][surface] / (data_dict['vertebrae'][top_vertebra]['AP_thickness']*data_dict['vertebrae'][top_vertebra]['median_thickness'])

            # Create asymmetry quotient
            if data_dict['foramens'][struc_name]['right_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != 0:
                data_dict['foramens'][struc_name]['asymmetry R/L'] = data_dict['foramens'][struc_name]['right_surface'] / data_dict['foramens'][struc_name]['left_surface']
            else:
                data_dict['foramens'][struc_name]['asymmetry R/L'] = -1
    return data_dict

def compute_vertebrae_metrics(data_dict):
    # Compute Vertebrae metrics
    for struc_name in data_dict['vertebrae'].keys():
        # Normalize foramen surfaces with top vertebra volume
        for metric in data_dict['vertebrae'][struc_name].keys():
            if metric != 'volume':
                data_dict['vertebrae'][struc_name][metric] = data_dict['vertebrae'][struc_name][metric] / data_dict['vertebrae'][struc_name]['volume']
    return data_dict

def rescale_canal(all_values):
    '''
    Rescale subject canals and CSF based on discs z coordinates.
    '''
    new_values = copy.deepcopy(all_values)
    for struc in ['canal', 'csf']:
        for struc_name in all_values[struc].keys():
            # Align all metrics for each subject using discs level as references
            disc_levels = all_values[struc][struc_name]['disc_level']
            # Flatten the list of arrays and concatenate all unique values
            all_discs = np.unique(np.concatenate([np.unique(dl) for dl in disc_levels]))
            all_discs = all_discs[~np.isnan(all_discs)]

            # For each subject count slices between discs
            n_subjects = len(disc_levels)
            gap_dict = {}
            for subj_idx in range(n_subjects):
                subj_disc_level = np.array(disc_levels[subj_idx])            
                subj_valid = ~pd.isna(subj_disc_level)
                subj_disc_positions = np.where(subj_valid)[0]
                subj_disc_values = subj_disc_level[subj_valid]

                # If the number of discs doesn't match, skip this subject
                if len(subj_disc_values) < 2:
                    continue
                
                # Create dict with number of slice between discs
                previous_disc = subj_disc_values[0]
                previous_pos = subj_disc_positions[0]
                for pos, disc in zip(subj_disc_positions[1:], subj_disc_values[1:]):
                    if f"{previous_disc}-{disc}" not in gap_dict:
                        gap_dict[f"{previous_disc}-{disc}"] = []
                    gap_dict[f"{previous_disc}-{disc}"].append(pos - previous_pos)
                    previous_disc = disc
                    previous_pos = pos

            # Pick max for each gap between discs in gap_dict
            for k, v in gap_dict.items():
                gap_dict[k] = int(round(np.median(v)))

            # Rescale subjects
            for subj_idx in range(n_subjects):
                for metric in all_values[struc][struc_name].keys():
                    if metric in ['slice_nb', 'disc_level']:
                        continue
                    interp_values, slice_interp = rescale_with_discs(disc_levels[subj_idx], all_values[struc][struc_name][metric][subj_idx], gap_dict)
                    new_values[struc][struc_name][metric][subj_idx] = interp_values
                if 'slice_interp' not in new_values[struc][struc_name]:
                    new_values[struc][struc_name]['slice_interp'] = []
                new_values[struc][struc_name]['slice_interp'].append(slice_interp)
                # Remove slice_nb and disc_level from dict
                new_values[struc][struc_name].pop('slice_nb', None)
                new_values[struc][struc_name].pop('disc_level', None)

            # Store gap_dict in all_values
            new_values[struc][struc_name]['discs_gap'] = gap_dict
    return new_values

def rescale_with_discs(disc_levels, metric_list, gap_dict):
    '''
    Return rescaled metric values and corresponding slice indices using disc levels and gap information.
    '''
    # Rescale data for each metric
    subj_disc_level = np.array(disc_levels)
    subj_valid = ~pd.isna(subj_disc_level)
    subj_disc_positions = np.where(subj_valid)[0]
    subj_disc_values = subj_disc_level[subj_valid]

    # If the number of discs doesn't match, skip this subject
    if len(subj_disc_values) < 2:
        return [], []

    # Rescale each metric with linear interpolation
    values = np.array(metric_list)
    interp_values = []
    slice_interp = []
    for disc_idx, disc in enumerate(subj_disc_values):
        if disc_idx < len(subj_disc_values) - 1:
            gap = gap_dict[f"{disc}-{subj_disc_values[disc_idx + 1]}"]
            yp = values[subj_disc_positions[disc_idx]:subj_disc_positions[disc_idx+1]]
            xp = np.linspace(0, gap-1, len(yp))
            x = np.linspace(0, gap-1, gap)
            if not -1 in yp:
                interp_func = np.interp(
                    x=x,
                    xp=xp,
                    fp=yp
                )
            else:
                interp_func = np.full_like(x, -1)
            interp_values += interp_func.tolist()

    start_disc_gap = 0
    k = list(gap_dict.keys())[0]
    i = 0
    while k != f"{subj_disc_values[0]}-{subj_disc_values[1]}":
        start_disc_gap += gap_dict[k]
        i += 1
        k = list(gap_dict.keys())[i]
    slice_interp += list(range(start_disc_gap, start_disc_gap + len(interp_values)))
    return interp_values, slice_interp

def create_dict_from_subject_data(subject_data, intensity_profile=True):
    """
    Create a dictionary from the subject data DataFrame.

    Parameters:
        subject_data (pd.DataFrame): The subject data DataFrame.

    Returns:
        dict: A dictionary with structure names as keys and DataFrames as values.
    """
    subject_dict = {}
    for struc in subject_data.name:
        struc_dict = {}
        struc_data = subject_data[subject_data['name'] == struc]
        struc_idx = struc_data.index[0]
        for column in struc_data.columns[2:]:
            if column != 'center':
                struc_dict[column] = struc_data[column][struc_idx]
        subject_dict[struc] = struc_dict
    return subject_dict

def convert_str_to_list(string):
    return [float(item.strip()) for item in string[1:-1].split(',')]

if __name__ == "__main__":
    main()