import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import copy
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, kruskal

def main():
    folder_path = Path('/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/metrics_output')
    demographics_path = Path('/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/lbp-lumbar-usf-2025/participants.tsv')
    output_folder = Path('images_by_group/')

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
            if struc not in ['spinalcord', 'canal', 'csf']:
                for struc_name in control_data[struc].keys():
                    if struc_name != 'spinalcord/canal':
                        for metric in control_data[struc][struc_name].keys():
                            # Add subject to all_values
                            subject_value = control_data[struc][struc_name][metric]
                            if subject_value != -1 and isinstance(sub_info['age'], (int, float)) and sub_info['sex'] in ['M', 'F']:
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
    # all_values = rescale_canal(all_values)

    # GROUP ANALYSIS AND ROBUSTNESS EVALUATION
    print("Starting demographic group analysis and robustness evaluation...")
    
    # Group data by demographics (age terciles and sex)
    grouped_data = group_data_by_demographics(all_values, all_demographics)
    
    # Calculate robustness metrics for subjects with multiple measurements
    robustness_data = calculate_robustness_metrics_by_group(grouped_data)
    
    # Create output folder
    output_folder.mkdir(exist_ok=True)
    
    # Generate plots by groups
    print("Generating plots by demographic groups...")
    plot_metrics_by_groups(grouped_data, robustness_data, output_folder)
    
    # Generate robustness summary
    print("Generating robustness analysis summary...")
    plot_robustness_summary(robustness_data, output_folder)
    
    # Perform statistical analysis
    print("Performing statistical analysis between groups...")
    perform_statistical_analysis(grouped_data, output_folder)
    
    print(f"Analysis complete! Results saved to {output_folder}")
    print(f"Check the following files:")
    print(f"  - Individual metric plots: {output_folder}/*_by_groups.png")
    print(f"  - Robustness summary: {output_folder}/robustness_summary.png")
    print(f"  - Statistical comparisons: {output_folder}/statistical_comparisons.csv")
    print(f"  - Robustness statistics: {output_folder}/robustness_summary_stats.csv")

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

def categorize_age_groups(age, method='terciles'):
    """
    Categorize age into groups.
    
    Args:
        age: Age value or list of ages
        method: 'terciles', 'decades', or 'custom'
    
    Returns:
        Age group label
    """
    if method == 'decades':
        if age < 30:
            return 'Young (18-29)'
        elif age < 50:
            return 'Middle (30-49)'
        elif age < 65:
            return 'Mature (50-64)'
        else:
            return 'Senior (65+)'
    elif method == 'custom':
        if age < 40:
            return 'Young (<40)'
        elif age < 60:
            return 'Middle (40-59)'
        else:
            return 'Older (60+)'
    else:  # terciles - will be determined from data
        return age  # Return raw age for later tercile calculation


def group_data_by_demographics(all_values, all_demographics):
    """
    Group data by age and sex for analysis.
    
    Args:
        all_values: Dictionary with structure {struc: {struc_name: {metric: [values]}}}
        all_demographics: Dictionary with structure {struc: {struc_name: {metric: [demographics]}}}
    
    Returns:
        Grouped data dictionary
    """
    grouped_data = {}
    sex_dict = {'M':'Male', 'F':'Female'}
    
    for struc in all_values.keys():
        grouped_data[struc] = {}
        
        for struc_name in all_values[struc].keys():
            grouped_data[struc][struc_name] = {}
            
            for metric in all_values[struc][struc_name].keys():
                if metric in ['discs_gap', 'slice_interp']:  # Skip non-metric data
                    continue
                    
                values = all_values[struc][struc_name][metric]
                demographics = all_demographics[struc][struc_name][metric]
                
                # Extract ages for tercile calculation
                ages = [demo['age'] for demo in demographics]
                age_terciles = np.percentile(ages, [33.33, 66.67])
                
                # Group data
                group_data = {
                    'Male_Young': {'values': [], 'demographics': []},
                    'Male_Middle': {'values': [], 'demographics': []},
                    'Male_Older': {'values': [], 'demographics': []},
                    'Female_Young': {'values': [], 'demographics': []},
                    'Female_Middle': {'values': [], 'demographics': []},
                    'Female_Older': {'values': [], 'demographics': []}
                }
                
                for val, demo in zip(values, demographics):
                    age = demo['age']
                    sex = sex_dict[demo['sex']]
                    
                    # Determine age group
                    if age <= age_terciles[0]:
                        age_group = 'Young'
                    elif age <= age_terciles[1]:
                        age_group = 'Middle'
                    else:
                        age_group = 'Older'
                    
                    # Create group key
                    group_key = f"{sex}_{age_group}"
                    
                    if group_key in group_data:
                        group_data[group_key]['values'].append(val)
                        group_data[group_key]['demographics'].append(demo)
                
                grouped_data[struc][struc_name][metric] = group_data
    
    return grouped_data


def calculate_robustness_metrics_by_group(grouped_data):
    """
    Calculate robustness metrics for subjects with multiple measurements.
    
    Args:
        grouped_data: Dictionary from group_data_by_demographics
    
    Returns:
        Dictionary with robustness metrics
    """
    robustness_data = {}
    
    for struc in grouped_data.keys():
        robustness_data[struc] = {}
        
        for struc_name in grouped_data[struc].keys():
            robustness_data[struc][struc_name] = {}
            
            for metric in grouped_data[struc][struc_name].keys():
                robustness_data[struc][struc_name][metric] = {}
                
                for group in grouped_data[struc][struc_name][metric].keys():
                    values = grouped_data[struc][struc_name][metric][group]['values']
                    demographics = grouped_data[struc][struc_name][metric][group]['demographics']
                    
                    # Group by subject (participant_id)
                    subject_values = {}
                    for val, demo in zip(values, demographics):
                        subj_id = demo['participant_id']
                        if subj_id not in subject_values:
                            subject_values[subj_id] = []
                        subject_values[subj_id].append(val)
                    
                    # Calculate robustness for subjects with multiple measurements
                    robustness_metrics = []
                    for subj_id, subj_vals in subject_values.items():
                        if len(subj_vals) > 1:  # Multiple measurements
                            subj_vals = np.array(subj_vals)
                            subj_vals = subj_vals[~np.isnan(subj_vals)]  # Remove NaN
                            
                            if len(subj_vals) > 1:
                                mean_val = np.mean(subj_vals)
                                std_val = np.std(subj_vals)
                                cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
                                
                                robustness_metrics.append({
                                    'subject': subj_id,
                                    'n_measurements': len(subj_vals),
                                    'mean': mean_val,
                                    'std': std_val,
                                    'cv_percent': cv,
                                    'values': subj_vals.tolist()
                                })
                    
                    robustness_data[struc][struc_name][metric][group] = robustness_metrics
    
    return robustness_data


def plot_metrics_by_groups(grouped_data, robustness_data, output_folder):
    """
    Create comprehensive plots showing metrics by age/sex groups with robustness analysis.
    
    Args:
        grouped_data: Dictionary from group_data_by_demographics
        robustness_data: Dictionary from calculate_robustness_metrics_by_group
        output_folder: Path to save plots
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    for struc in grouped_data.keys():
        for struc_name in grouped_data[struc].keys():
            metrics = [m for m in grouped_data[struc][struc_name].keys() 
                      if m not in ['discs_gap', 'slice_interp']]
            
            if not metrics:
                continue
            
            # Create subplot grid
            n_metrics = len(metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx] if n_metrics > 1 else axes[0]
                
                # Prepare data for plotting
                plot_data = []
                group_names = []
                colors = []
                
                # Define colors for groups
                color_map = {
                    'Male_Young': "#1B6AA2",
                    'Male_Middle': '#ff7f0e', 
                    'Male_Older': '#2ca02c',
                    'Female_Young': '#d62728',
                    'Female_Middle': '#9467bd',
                    'Female_Older': '#8c564b'
                }
                
                for group in ['Male_Young', 'Male_Middle', 'Male_Older', 
                             'Female_Young', 'Female_Middle', 'Female_Older']:
                    values = grouped_data[struc][struc_name][metric][group]['values']
                    if values:
                        plot_data.append(values)
                        group_names.append(group.replace('_', '\n'))
                        colors.append(color_map[group])
                
                if plot_data:
                    # Box plot
                    bp = ax.boxplot(plot_data, labels=group_names, patch_artist=True)
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Add individual points with jitter
                    for i, (data, group) in enumerate(zip(plot_data, group_names)):
                        x = np.random.normal(i+1, 0.04, len(data))
                        ax.scatter(x, data, alpha=0.6, s=30, color=colors[i])
                    
                    ax.set_title(f'{metric}\n{struc_name} ({struc})')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add robustness info as text
                    robustness_info = robustness_data[struc][struc_name][metric]
                    cv_means = []
                    for group in ['Male_Young', 'Male_Middle', 'Male_Older', 
                                 'Female_Young', 'Female_Middle', 'Female_Older']:
                        if robustness_info[group]:
                            cvs = [r['cv_percent'] for r in robustness_info[group] 
                                  if not np.isnan(r['cv_percent'])]
                            if cvs:
                                cv_means.append(np.mean(cvs))
                            else:
                                cv_means.append(np.nan)
                        else:
                            cv_means.append(np.nan)
                    
                    # Add text box with robustness info
                    valid_cvs = [cv for cv in cv_means if not np.isnan(cv)]
                    if valid_cvs:
                        ax.text(0.02, 0.98, f'Mean CV: {np.mean(valid_cvs):.1f}%', 
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top', bbox=dict(boxstyle='round', 
                               facecolor='wheat', alpha=0.5))
            
            # Remove empty subplots
            for idx in range(n_metrics, len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(output_folder / f'{struc}_{struc_name}_by_groups.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot for {struc} - {struc_name}")


def plot_robustness_summary(robustness_data, output_folder):
    """
    Create summary plots for robustness analysis across all metrics and groups.
    
    Args:
        robustness_data: Dictionary from calculate_robustness_metrics_by_group
        output_folder: Path to save plots
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Collect all CV values
    all_cv_data = []
    
    for struc in robustness_data.keys():
        for struc_name in robustness_data[struc].keys():
            for metric in robustness_data[struc][struc_name].keys():
                for group in robustness_data[struc][struc_name][metric].keys():
                    for subject_data in robustness_data[struc][struc_name][metric][group]:
                        if not np.isnan(subject_data['cv_percent']):
                            all_cv_data.append({
                                'structure': struc,
                                'structure_name': struc_name,
                                'metric': metric,
                                'group': group,
                                'cv_percent': subject_data['cv_percent'],
                                'n_measurements': subject_data['n_measurements'],
                                'subject': subject_data['subject']
                            })
    
    if not all_cv_data:
        print("No robustness data available for plotting")
        return
    
    cv_df = pd.DataFrame(all_cv_data)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. CV distribution by structure
    ax = axes[0, 0]
    structures = cv_df['structure'].unique()
    cv_by_struct = [cv_df[cv_df['structure'] == s]['cv_percent'].values for s in structures]
    ax.boxplot(cv_by_struct, labels=structures)
    ax.set_title('CV Distribution by Structure')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.grid(True, alpha=0.3)
    
    # 2. CV distribution by group
    ax = axes[0, 1]
    groups = cv_df['group'].unique()
    cv_by_group = [cv_df[cv_df['group'] == g]['cv_percent'].values for g in groups]
    ax.boxplot(cv_by_group, labels=[g.replace('_', '\n') for g in groups])
    ax.set_title('CV Distribution by Demographic Group')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 3. Number of subjects with multiple measurements
    ax = axes[0, 2]
    subjects_per_group = cv_df.groupby('group')['subject'].nunique()
    bars = ax.bar(range(len(subjects_per_group)), subjects_per_group.values)
    ax.set_xticks(range(len(subjects_per_group)))
    ax.set_xticklabels([g.replace('_', '\n') for g in subjects_per_group.index], rotation=45)
    ax.set_title('Subjects with Multiple Measurements')
    ax.set_ylabel('Number of Subjects')
    ax.grid(True, alpha=0.3)
    
    # 4. CV vs Number of measurements
    ax = axes[1, 0]
    ax.scatter(cv_df['n_measurements'], cv_df['cv_percent'], alpha=0.6)
    ax.set_xlabel('Number of Measurements')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('CV vs Number of Measurements')
    ax.grid(True, alpha=0.3)
    
    # 5. Heatmap of mean CV by structure and group
    ax = axes[1, 1]
    pivot_cv = cv_df.groupby(['structure', 'group'])['cv_percent'].mean().unstack(fill_value=np.nan)
    sns.heatmap(pivot_cv, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax)
    ax.set_title('Mean CV by Structure and Group')
    ax.set_xlabel('Demographic Group')
    ax.set_ylabel('Structure')
    
    # 6. Distribution of high/low variability subjects
    ax = axes[1, 2]
    cv_df['variability_category'] = pd.cut(cv_df['cv_percent'], 
                                          bins=[0, 5, 20, 100], 
                                          labels=['Low (<5%)', 'Medium (5-20%)', 'High (>20%)'])
    variability_counts = cv_df['variability_category'].value_counts()
    ax.pie(variability_counts.values, labels=variability_counts.index, autopct='%1.1f%%')
    ax.set_title('Distribution of Measurement Variability')
    
    plt.tight_layout()
    plt.savefig(output_folder / 'robustness_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary_stats = cv_df.groupby(['structure', 'group']).agg({
        'cv_percent': ['count', 'mean', 'median', 'std'],
        'subject': 'nunique'
    }).round(3)
    
    summary_stats.to_csv(output_folder / 'robustness_summary_stats.csv')
    print(f"Robustness summary saved to {output_folder}")


def perform_statistical_analysis(grouped_data, output_folder):
    """
    Perform statistical tests between groups.
    
    Args:
        grouped_data: Dictionary from group_data_by_demographics
        output_folder: Path to save results
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    results = []
    
    for struc in grouped_data.keys():
        for struc_name in grouped_data[struc].keys():
            for metric in grouped_data[struc][struc_name].keys():
                if metric in ['discs_gap', 'slice_interp']:
                    continue
                
                # Get values for each group
                group_values = {}
                for group in grouped_data[struc][struc_name][metric].keys():
                    values = grouped_data[struc][struc_name][metric][group]['values']
                    if values:
                        group_values[group] = values
                
                # Perform pairwise comparisons
                groups = list(group_values.keys())
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        group1, group2 = groups[i], groups[j]
                        if len(group_values[group1]) > 5 and len(group_values[group2]) > 5:
                            statistic, p_value = mannwhitneyu(group_values[group1], 
                                                            group_values[group2])
                            
                            results.append({
                                'structure': struc,
                                'structure_name': struc_name,
                                'metric': metric,
                                'group1': group1,
                                'group2': group2,
                                'group1_mean': np.mean(group_values[group1]),
                                'group2_mean': np.mean(group_values[group2]),
                                'group1_n': len(group_values[group1]),
                                'group2_n': len(group_values[group2]),
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_folder / 'statistical_comparisons.csv', index=False)
        
        # Summary of significant differences
        significant_results = results_df[results_df['significant']]
        print(f"Found {len(significant_results)} significant differences out of {len(results)} comparisons")
        
        if len(significant_results) > 0:
            print("\nTop 10 most significant differences:")
            top_significant = significant_results.nsmallest(10, 'p_value')
            print(top_significant[['structure', 'structure_name', 'metric', 'group1', 'group2', 'p_value']].to_string(index=False))


def convert_str_to_list(string):
    return [float(item.strip()) for item in string[1:-1].split(',')]


if __name__ == "__main__":
    main()