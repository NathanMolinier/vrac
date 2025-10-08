import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    folder_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/metrics_output'
    demographics_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/lbp-lumbar-usf-2025/participants.tsv'

    demographics = pd.read_csv(demographics_path, sep='\t')

    thickness_dict = {}
    intensity_dict = {}
    solidity_dict = {}
    volume_dict = {}
    eccentricity_dict = {}
    img_dict = {}
    seg_dict = {}
    age_dict = {}
    filename_dict = {}
    sex_dict = {}
    filter = ''
    for sub in os.listdir(folder_path):
        csv_folder = os.path.join(folder_path, sub, "csv")
        discs_imgs = os.path.join(folder_path, sub, "imgs")
        sub_name = sub.split('_')[0]
        sub_info = demographics[demographics['participant_id'] == sub_name]
        if os.path.exists(csv_folder) and filter in sub:
            discs_data = pd.read_csv(os.path.join(csv_folder, "discs.csv"))
            vertebrae_data = pd.read_csv(os.path.join(csv_folder, "vertebrae.csv"))

            for name, intensity_peaks_gap, thickness, solidity, eccentricity, volume in zip(discs_data.name, discs_data.intensity_peaks_gap, discs_data.median_thickness, discs_data.solidity, discs_data.eccentricity, discs_data.volume):
                if name not in intensity_dict:
                    intensity_dict[name] = {}
                    thickness_dict[name] = {}
                    solidity_dict[name] = {}
                    eccentricity_dict[name] = {}
                    volume_dict[name] = {}
                    img_dict[name] = {}
                    seg_dict[name] = {}
                    age_dict[name] = {}
                    sex_dict[name] = {}
                    filename_dict[name] = {}

                # Find in dataframe overlying_vert in name
                overlying_vert = name.split('-')[0]
                matching_rows = vertebrae_data[vertebrae_data['name'] == overlying_vert]
                age = sub_info['age'].iloc[0]
                if not matching_rows.empty and (isinstance(age, float) or isinstance(age, int)):
                    if sub_name not in thickness_dict[name].keys():
                        thickness_dict[name][sub_name] = []
                        intensity_dict[name][sub_name] = []
                        solidity_dict[name][sub_name] = []
                        eccentricity_dict[name][sub_name] = []
                        volume_dict[name][sub_name] = []
                        img_dict[name][sub_name] = []
                        seg_dict[name][sub_name] = []
                        age_dict[name][sub_name] = []
                        sex_dict[name][sub_name] = []
                        filename_dict[name][sub_name] = []
                    ap_thickness = float(vertebrae_data[vertebrae_data['name'] == overlying_vert]['AP_thickness'].iloc[0])
                    thickness_dict[name][sub_name].append(thickness/ap_thickness)
                    intensity_dict[name][sub_name].append(intensity_peaks_gap)

                    # Add solidity and eccentricity
                    solidity_dict[name][sub_name].append(solidity)
                    eccentricity_dict[name][sub_name].append(eccentricity)

                    # Add volume normalized by vertebrae volume
                    vert_volume = float(vertebrae_data[vertebrae_data['name'] == overlying_vert]['volume'].iloc[0])
                    volume_dict[name][sub_name].append(volume/vert_volume)

                    # Add image
                    img_dict[name][sub_name].append(np.rot90(plt.imread(os.path.join(discs_imgs, f'discs_{name}_img.png'))))
                    seg_dict[name][sub_name].append(np.rot90(plt.imread(os.path.join(discs_imgs, f'discs_{name}_seg.png'))))

                    # File names
                    filename_dict[name][sub_name].append(sub)

                    # Add age and sex
                    age_dict[name][sub_name].append(age)
                    sex_dict[name][sub_name].append(sub_info['sex'].iloc[0])

    # ROBUSTNESS ANALYSIS
    print("Starting robustness analysis...")
    
    # Create dictionary of all metrics for analysis
    all_metrics = {
        'thickness': thickness_dict,
        'intensity': intensity_dict,
        'solidity': solidity_dict,
        'eccentricity': eccentricity_dict,
        'volume': volume_dict
    }
    
    # Generate comprehensive robustness analysis
    generate_robustness_summary(all_metrics)
    
    # Create detailed robustness plots for each metric
    for metric_name, metric_dict in all_metrics.items():
        print(f"Analyzing robustness for {metric_name}...")
        plot_robustness_analysis(metric_dict, metric_name)
        plot_side_by_side_comparison(metric_dict, metric_name)
    
    # Create correlation analysis
    plot_correlation_matrix(all_metrics)
    
    print("Robustness analysis complete! Check the 'imgs' folder for all outputs.")


def calculate_robustness_metrics(metric_dict):
    """
    Calculate robustness metrics for each subject and disc level.
    
    Args:
        metric_dict: Dictionary with structure {disc_name: {subject: [values]}}
        
    Returns:
        DataFrame with robustness metrics per subject and disc
    """
    robustness_data = []
    
    for disc_name, disc_data in metric_dict.items():
        for subject, values in disc_data.items():
            if len(values) > 1:  # Only calculate for subjects with multiple measurements
                values = np.array(values)
                # Remove any NaN values
                values = values[~np.isnan(values)]
                
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan  # Coefficient of variation as percentage
                    min_val = np.min(values)
                    max_val = np.max(values)
                    range_val = max_val - min_val
                    relative_range = (range_val / mean_val) * 100 if mean_val != 0 else np.nan
                    
                    robustness_data.append({
                        'disc': disc_name,
                        'subject': subject,
                        'n_measurements': len(values),
                        'mean': mean_val,
                        'std': std_val,
                        'cv_percent': cv,
                        'min': min_val,
                        'max': max_val,
                        'range': range_val,
                        'relative_range_percent': relative_range,
                        'measurements': values.tolist()
                    })
    
    return pd.DataFrame(robustness_data)


def analyze_measurement_agreement(metric_dict, metric_name):
    """
    Analyze agreement between multiple measurements using Bland-Altman style analysis.
    
    Args:
        metric_dict: Dictionary with structure {disc_name: {subject: [values]}}
        metric_name: Name of the metric for labeling
        
    Returns:
        DataFrame with pairwise comparisons
    """
    comparison_data = []
    
    for disc_name, disc_data in metric_dict.items():
        for subject, values in disc_data.items():
            if len(values) >= 2:  # Need at least 2 measurements
                values = np.array(values)
                values = values[~np.isnan(values)]
                
                # For each pair of measurements
                for i in range(len(values)):
                    for j in range(i+1, len(values)):
                        val1, val2 = values[i], values[j]
                        mean_pair = (val1 + val2) / 2
                        diff_pair = val1 - val2
                        
                        comparison_data.append({
                            'disc': disc_name,
                            'subject': subject,
                            'measurement_1': val1,
                            'measurement_2': val2,
                            'mean': mean_pair,
                            'difference': diff_pair,
                            'abs_difference': abs(diff_pair),
                            'relative_difference_percent': (diff_pair / mean_pair) * 100 if mean_pair != 0 else np.nan
                        })
    
    return pd.DataFrame(comparison_data)


def plot_robustness_analysis(metric_dict, metric_name, output_dir='imgs'):
    """
    Create comprehensive robustness visualization plots.
    
    Args:
        metric_dict: Dictionary with structure {disc_name: {subject: [values]}}
        metric_name: Name of the metric for labeling
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate robustness metrics
    robustness_df = calculate_robustness_metrics(metric_dict)
    
    if robustness_df.empty:
        print(f"No subjects with multiple measurements found for {metric_name}")
        return
    
    # 1. Coefficient of Variation by Disc Level
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    disc_order = sorted(robustness_df['disc'].unique())
    cv_data = [robustness_df[robustness_df['disc'] == disc]['cv_percent'].dropna().values 
               for disc in disc_order]
    
    plt.boxplot(cv_data, labels=disc_order)
    plt.xticks(rotation=45)
    plt.ylabel('Coefficient of Variation (%)')
    plt.title(f'{metric_name} - CV by Disc Level')
    plt.grid(True, alpha=0.3)
    
    # 2. Individual Subject Measurements
    plt.subplot(1, 3, 2)
    subjects_multi = robustness_df['subject'].unique()[:10]  # Show first 10 subjects
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects_multi)))
    
    for i, subject in enumerate(subjects_multi):
        subject_data = robustness_df[robustness_df['subject'] == subject]
        if not subject_data.empty:
            for _, row in subject_data.iterrows():
                measurements = row['measurements']
                x_pos = [i] * len(measurements)
                plt.scatter(x_pos, measurements, color=colors[i], alpha=0.7, s=50)
                if len(measurements) > 1:
                    plt.plot([i, i], [min(measurements), max(measurements)], 
                            color=colors[i], alpha=0.5, linewidth=2)
    
    plt.xticks(range(len(subjects_multi)), subjects_multi, rotation=45)
    plt.ylabel(f'{metric_name}')
    plt.title(f'Individual Measurements per Subject')
    plt.grid(True, alpha=0.3)
    
    # 3. Bland-Altman style plot
    plt.subplot(1, 3, 3)
    comparison_df = analyze_measurement_agreement(metric_dict, metric_name)
    
    if not comparison_df.empty:
        plt.scatter(comparison_df['mean'], comparison_df['difference'], alpha=0.6)
        
        # Add mean difference line and limits of agreement
        mean_diff = comparison_df['difference'].mean()
        std_diff = comparison_df['difference'].std()
        
        plt.axhline(mean_diff, color='red', linestyle='-', label=f'Mean diff: {mean_diff:.3f}')
        plt.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', 
                   label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.3f}')
        plt.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', 
                   label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.3f}')
        
        plt.xlabel(f'Mean {metric_name}')
        plt.ylabel(f'Difference {metric_name}')
        plt.title('Measurement Agreement')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric_name}_robustness_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Robustness analysis plot saved for {metric_name}")


def plot_side_by_side_comparison(metric_dict, metric_name, output_dir='imgs'):
    """
    Create side-by-side comparison plots for subjects with multiple measurements.
    
    Args:
        metric_dict: Dictionary with structure {disc_name: {subject: [values]}}
        metric_name: Name of the metric for labeling
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find subjects with multiple measurements
    subjects_with_multiple = []
    for disc_name, disc_data in metric_dict.items():
        for subject, values in disc_data.items():
            if len(values) > 1:
                subjects_with_multiple.append((subject, disc_name, values))
    
    if not subjects_with_multiple:
        print(f"No subjects with multiple measurements found for {metric_name}")
        return
    
    # Group by subject
    subject_groups = {}
    for subject, disc, values in subjects_with_multiple:
        if subject not in subject_groups:
            subject_groups[subject] = {}
        subject_groups[subject][disc] = values
    
    # Create plots for each subject (show up to 6 subjects per figure)
    subjects = list(subject_groups.keys())
    n_subjects_per_fig = 6
    
    for fig_idx in range(0, len(subjects), n_subjects_per_fig):
        fig_subjects = subjects[fig_idx:fig_idx + n_subjects_per_fig]
        
        plt.figure(figsize=(20, 12))
        
        for i, subject in enumerate(fig_subjects):
            plt.subplot(2, 3, i + 1)
            
            discs = list(subject_groups[subject].keys())
            disc_positions = range(len(discs))
            
            # Plot all measurements for each disc
            for j, disc in enumerate(discs):
                values = subject_groups[subject][disc]
                x_pos = [j] * len(values)
                
                # Add some jitter to x position for visibility
                x_jitter = np.random.normal(0, 0.05, len(values))
                x_pos_jittered = [j + jit for jit in x_jitter]
                
                plt.scatter(x_pos_jittered, values, alpha=0.7, s=60)
                
                # Connect measurements with a line if more than one
                if len(values) > 1:
                    plt.plot([j, j], [min(values), max(values)], 
                            color='gray', alpha=0.5, linewidth=2)
                
                # Add mean line
                mean_val = np.mean(values)
                plt.plot([j-0.2, j+0.2], [mean_val, mean_val], 
                        color='red', linewidth=3, alpha=0.8)
            
            plt.xticks(disc_positions, discs, rotation=45)
            plt.ylabel(f'{metric_name}')
            plt.title(f'Subject: {subject}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric_name}_side_by_side_comparison_fig{fig_idx//n_subjects_per_fig + 1}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Side-by-side comparison plot {fig_idx//n_subjects_per_fig + 1} saved for {metric_name}")
        break


def plot_correlation_matrix(all_metrics_dict, output_dir='imgs'):
    """
    Plot correlation matrix between different metrics using repeated measurements.
    
    Args:
        all_metrics_dict: Dictionary containing all metric dictionaries
        output_dir: Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect all measurements in a single DataFrame
    all_data = []
    
    for metric_name, metric_dict in all_metrics_dict.items():
        for disc_name, disc_data in metric_dict.items():
            for subject, values in disc_data.items():
                for i, value in enumerate(values):
                    all_data.append({
                        'metric': metric_name,
                        'disc': disc_name,
                        'subject': subject,
                        'measurement_idx': i,
                        'value': value,
                        'subject_disc': f"{subject}_{disc_name}"
                    })
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No data available for correlation analysis")
        return
    
    # Create pivot table for correlation analysis
    pivot_df = df.pivot_table(index=['subject_disc', 'measurement_idx'], 
                             columns='metric', values='value', aggfunc='first')
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    plt.title('Correlation Matrix Between Metrics\n(Using All Repeated Measurements)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Correlation matrix plot saved")


def generate_robustness_summary(all_metrics_dict, output_dir='imgs'):
    """
    Generate comprehensive summary statistics for robustness analysis.
    
    Args:
        all_metrics_dict: Dictionary containing all metric dictionaries
        output_dir: Directory to save summary files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    summary_data = []
    
    for metric_name, metric_dict in all_metrics_dict.items():
        robustness_df = calculate_robustness_metrics(metric_dict)
        
        if not robustness_df.empty:
            # Overall statistics for this metric
            overall_stats = {
                'metric': metric_name,
                'n_subjects_with_multiple': len(robustness_df),
                'total_measurements': robustness_df['n_measurements'].sum(),
                'mean_cv_percent': robustness_df['cv_percent'].mean(),
                'median_cv_percent': robustness_df['cv_percent'].median(),
                'std_cv_percent': robustness_df['cv_percent'].std(),
                'mean_relative_range_percent': robustness_df['relative_range_percent'].mean(),
                'median_relative_range_percent': robustness_df['relative_range_percent'].median(),
                'subjects_with_high_variability_cv_20': (robustness_df['cv_percent'] > 20).sum(),
                'subjects_with_low_variability_cv_5': (robustness_df['cv_percent'] < 5).sum()
            }
            summary_data.append(overall_stats)
            
            # Save detailed robustness data for this metric
            # robustness_df.to_csv(os.path.join(output_dir, f'{metric_name}_detailed_robustness.csv'), index=False)
            
            # By disc level summary
            disc_summary = robustness_df.groupby('disc').agg({
                'cv_percent': ['count', 'mean', 'median', 'std'],
                'relative_range_percent': ['mean', 'median', 'std'],
                'n_measurements': 'mean'
            }).round(3)
            
            disc_summary.columns = ['_'.join(col).strip() for col in disc_summary.columns]
            # disc_summary.to_csv(os.path.join(output_dir, f'{metric_name}_disc_level_robustness.csv'))
            
            print(f"Detailed robustness analysis saved for {metric_name}")
    
    # Create overall summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(3)
        # summary_df.to_csv(os.path.join(output_dir, 'overall_robustness_summary.csv'), index=False)
        
        # Print summary to console
        print("\n" + "="*80)
        print("ROBUSTNESS ANALYSIS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("\n" + "="*80)
        
        # Create summary visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: CV comparison across metrics
        plt.subplot(2, 3, 1)
        metrics = summary_df['metric']
        cv_means = summary_df['mean_cv_percent']
        plt.bar(metrics, cv_means)
        plt.xticks(rotation=45)
        plt.ylabel('Mean CV (%)')
        plt.title('Mean Coefficient of Variation by Metric')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Number of subjects with multiple measurements
        plt.subplot(2, 3, 2)
        plt.bar(metrics, summary_df['n_subjects_with_multiple'])
        plt.xticks(rotation=45)
        plt.ylabel('Number of Subjects')
        plt.title('Subjects with Multiple Measurements')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: High vs Low variability subjects
        plt.subplot(2, 3, 3)
        x = np.arange(len(metrics))
        width = 0.35
        plt.bar(x - width/2, summary_df['subjects_with_high_variability_cv_20'], 
                width, label='High variability (CV>20%)', alpha=0.7)
        plt.bar(x + width/2, summary_df['subjects_with_low_variability_cv_5'], 
                width, label='Low variability (CV<5%)', alpha=0.7)
        plt.xticks(x, metrics, rotation=45)
        plt.ylabel('Number of Subjects')
        plt.title('Variability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: CV distribution across metrics
        plt.subplot(2, 3, 4)
        cv_data = []
        metric_labels = []
        for metric_name, metric_dict in all_metrics_dict.items():
            robustness_df = calculate_robustness_metrics(metric_dict)
            if not robustness_df.empty:
                cv_data.append(robustness_df['cv_percent'].dropna().values)
                metric_labels.append(metric_name)
        
        if cv_data:
            plt.boxplot(cv_data, labels=metric_labels)
            plt.xticks(rotation=45)
            plt.ylabel('CV (%)')
            plt.title('CV Distribution by Metric')
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Relative range distribution
        plt.subplot(2, 3, 5)
        range_data = []
        for metric_name, metric_dict in all_metrics_dict.items():
            robustness_df = calculate_robustness_metrics(metric_dict)
            if not robustness_df.empty:
                range_data.append(robustness_df['relative_range_percent'].dropna().values)
        
        if range_data:
            plt.boxplot(range_data, labels=metric_labels)
            plt.xticks(rotation=45)
            plt.ylabel('Relative Range (%)')
            plt.title('Relative Range Distribution by Metric')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Mean number of measurements per subject
        plt.subplot(2, 3, 6)
        plt.bar(metrics, summary_df['total_measurements'] / summary_df['n_subjects_with_multiple'])
        plt.xticks(rotation=45)
        plt.ylabel('Mean Measurements per Subject')
        plt.title('Average Measurements per Subject')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'robustness_summary_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plots saved to {output_dir}/robustness_summary_plots.png")
        print(f"Summary table saved to {output_dir}/overall_robustness_summary.csv")


def line(x, a, b):
    return a * x + b

def exp(x, a, b):
    return b + np.exp(a * x)

def convert_str_to_list(string):
    return [float(item) for item in string[1:-1].split(',')]

if __name__ == "__main__":
    main()