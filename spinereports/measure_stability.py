import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_plots(df_all, numeric_cols, file_name, plots_dir):
    """Generate Bland-Altman and Scatter plots for numeric columns."""
    icc_results = {}
    for col in numeric_cols:
        col_t1 = f"{col}_T1w"
        col_t2 = f"{col}_T2w"
        
        if col_t1 not in df_all.columns or col_t2 not in df_all.columns:
            continue
            
        val_t1 = pd.to_numeric(df_all[col_t1], errors='coerce')
        val_t2 = pd.to_numeric(df_all[col_t2], errors='coerce')
        
        # Drop NaNs and -1 values
        valid_idx = val_t1.notna() & val_t2.notna() & (val_t1 != -1) & (val_t2 != -1)
        v1 = val_t1[valid_idx]
        v2 = val_t2[valid_idx]
        
        if len(v1) == 0:
            continue
            
        icc_val = calculate_icc(v1.values, v2.values)
        if not np.isnan(icc_val):
            icc_results[col] = icc_val
            
        # Create a structure-specific folder
        struct_dir = plots_dir / file_name.replace('.csv', '')
        struct_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Scatter Plot
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=v1, y=v2, alpha=0.6)
        min_val = min(v1.min(), v2.min())
        max_val = max(v1.max(), v2.max())
        if not np.isnan(min_val) and not np.isnan(max_val):
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (Perfect Agreement)')
        plt.title(f'Scatter Plot - {col}\n({file_name})')
        plt.xlabel('T1w')
        plt.ylabel('T2w')
        plt.legend()
        plt.tight_layout()
        plt.savefig(struct_dir / f"scatter_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bland-Altman Plot
        mean_vals = (v1 + v2) / 2
        diff_vals = v1 - v2
        md = np.mean(diff_vals)
        sd = np.std(diff_vals, axis=0)
        
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=mean_vals, y=diff_vals, alpha=0.6)
        plt.axhline(md, color='red', linestyle='-', label=f'Mean Bias: {md:.2f}')
        plt.axhline(md + 1.96*sd, color='blue', linestyle='--', label=f'+1.96 SD: {md + 1.96*sd:.2f}')
        plt.axhline(md - 1.96*sd, color='blue', linestyle='--', label=f'-1.96 SD: {md - 1.96*sd:.2f}')
        plt.axhline(0, color='gray', linestyle=':')
        plt.title(f'Bland-Altman Plot - {col}\n({file_name})')
        plt.xlabel('Average of T1w and T2w')
        plt.ylabel('Difference (T1w - T2w)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(struct_dir / f"bland_altman_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Bar Plot for ICC Summary across all metrics in this file
    if icc_results:
        plt.figure(figsize=(8, max(5, len(icc_results) * 0.4)))
        sorted_iccs = sorted(icc_results.items(), key=lambda item: item[1])
        metrics = [item[0] for item in sorted_iccs]
        values = [item[1] for item in sorted_iccs]
        
        sns.barplot(x=values, y=metrics, hue=metrics, palette='viridis', legend=False)
        plt.axvline(1.0, color='black')
        plt.axvline(0.9, color='green', linestyle=':', label='Excellent (>0.9)')
        plt.axvline(0.75, color='orange', linestyle=':', label='Good (>0.75)')
        plt.xlim(0, 1.05)
        plt.title(f'Intraclass Correlation Coefficients (ICC)\n{file_name}')
        plt.xlabel('ICC(1,1)')
        plt.ylabel('Metric')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(struct_dir / "icc_summary_barplot.png", dpi=300, bbox_inches='tight')
        plt.close()

def calculate_icc(v1, v2):
    """Calculate Intraclass Correlation Coefficient (ICC) - one-way random effects."""
    n = len(v1)
    if n < 2: 
        return np.nan
    k = 2
    mean_subj = (v1 + v2) / 2
    grand_mean = np.mean(mean_subj)
    
    SSB = k * np.sum((mean_subj - grand_mean)**2)
    MSB = SSB / (n - 1)
    
    SSW = np.sum((v1 - mean_subj)**2 + (v2 - mean_subj)**2)
    MSW = SSW / n
    
    if (MSB + (k - 1) * MSW) == 0:
        return np.nan
        
    icc = (MSB - MSW) / (MSB + (k - 1) * MSW)
    return icc

def calculate_stability(df, numeric_cols, suffix1='_T1w', suffix2='_T2w'):
    """Calculate Mean Absolute Difference, Mean Relative Difference, and Correlation."""
    stability_results = []
    
    for col in numeric_cols:
        col_t1 = f"{col}{suffix1}"
        col_t2 = f"{col}{suffix2}"
        
        # Ensure both columns are in the merged dataframe
        if col_t1 not in df.columns or col_t2 not in df.columns:
            continue
            
        val_t1 = pd.to_numeric(df[col_t1], errors='coerce')
        val_t2 = pd.to_numeric(df[col_t2], errors='coerce')
        
        # Drop NaNs and -1 values for valid pairwise comparison
        valid_idx = val_t1.notna() & val_t2.notna() & (val_t1 != -1) & (val_t2 != -1)
        v1 = val_t1[valid_idx]
        v2 = val_t2[valid_idx]
        
        if len(v1) == 0:
            continue
            
        abs_diff = np.abs(v2 - v1)
        rel_diff = abs_diff / (np.abs(v1) + 1e-9) # avoid division by zero
        
        corr = v1.corr(v2) if len(v1) > 1 else np.nan
        icc_val = calculate_icc(v1.values, v2.values)
        
        stability_results.append({
            'Metric': col,
            'Mean_Absolute_Diff': abs_diff.mean(),
            'Mean_Relative_Diff_Pct': rel_diff.mean() * 100,
            'Pearson_Correlation': corr,
            'ICC': icc_val,
            'N_Samples': len(v1)
        })
        
    return pd.DataFrame(stability_results)

def main():
    # parser = argparse.ArgumentParser(description="Evaluate stability of metrics between T1w and T2w.")
    # parser.add_argument('-i', '--input', required=True, help="Input directory containing subject folders")
    # parser.add_argument('-o', '--output', default='stability_report.csv', help="Output CSV for the aggregated stability report")
    # args = parser.parse_args()

    input_dir = Path("/home/ge.polymtl.ca/p118739/data/datasets/analysis_balgrist/reports_t1w_t2w") #Path(args.input)
    output_dir = Path("/home/ge.polymtl.ca/p118739/data/datasets/analysis_balgrist/reports_t1w_t2w") #Path(args.output)
    
    # Identify unique subjects by crawling sub-* folders
    all_dirs = [d.name for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    
    subjects = set()
    for d in all_dirs:
        # Get base subject identifier
        parts = d.split('_')
        base_sub = '_'.join(parts[:-1]) # e.g., sub-001_acq-sag
        subjects.add(base_sub)

    # Configuration for alignment strategies
    alignment_rules = {
        'discs_subject.csv': ['structure_name'],
        'vertebrae_subject.csv': ['structure_name'],
        'foramens_subject.csv': ['structure_name'],
        'canal_subject.csv': ['structure_name', 'slice_interp'],
        'csf_subject.csv': ['structure_name', 'slice_interp']
    }
    
    # Store aggregated merged data across all subjects
    aggregated_data = {file: [] for file in alignment_rules.keys()}

    # 1. Iterate and aggregate paired data
    for sub in subjects:
        # Reconstruct actual folder names
        t1_dir = input_dir / f"{sub}_T1w" / "files"
        t2_dir = input_dir / f"{sub}_T2w" / "files"
        
        if not (t1_dir.exists() and t2_dir.exists()):
            print(f"Skipping {sub}: Missing either T1w or T2w files folder.")
            continue
            
        for file_name, merge_keys in alignment_rules.items():
            t1_file = t1_dir / file_name
            t2_file = t2_dir / file_name
            
            if t1_file.exists() and t2_file.exists():
                df_t1 = pd.read_csv(t1_file)
                df_t2 = pd.read_csv(t2_file)
                
                # Merge T1w and T2w on defined keys
                df_merged = pd.merge(
                    df_t1, 
                    df_t2, 
                    on=merge_keys, 
                    suffixes=('_T1w', '_T2w'),
                    how='inner'
                )
                df_merged['subject'] = sub
                aggregated_data[file_name].append(df_merged)

    # 2. Evaluate stability across the entire aggregated dataset
    final_reports = []
    
    for file_name, file_data in aggregated_data.items():
        if not file_data:
            continue
            
        df_all = pd.concat(file_data, ignore_index=True)
        
        exclude_cols = ['structure', 'structure_name', 'slice_interp', 'vertebra_level', 'subject', 'nucleus_eccentricity_AP-RL', 
                        'nucleus_eccentricity_AP-SI','nucleus_eccentricity_RL-SI','nucleus_solidity','nucleus_volume',
                        'nucleus_median_thickness','intensity_variation','median_signal','ap_attenuation','left_compression_ratio',
                        'right_compression_ratio','slice_signal','left_slice_signal','right_slice_signal']
        
        if file_name == 'canal_subject.csv':
            exclude_cols += ['asymmetry_R_L', 'right_area', 'left_area']
        elif file_name == 'discs_subject.csv':
            exclude_cols += ['eccentricity']

        numeric_cols = [c.replace('_T1w', '') for c in df_all.columns if c.endswith('_T1w') and not any(ext in c for ext in exclude_cols)]
        
        # Calculate stability
        stability_df = calculate_stability(df_all, numeric_cols)
        stability_df['File'] = file_name
        final_reports.append(stability_df)

        # Generate plots
        plots_dir = output_dir / "out_stability"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generate_plots(df_all, numeric_cols, file_name, plots_dir)

    if final_reports:
        # 3. Export global structural stability report
        final_report_df = pd.concat(final_reports, ignore_index=True)
        
        # Reorder columns for readability
        cols = ['File', 'Metric', 'N_Samples', 'ICC', 'Pearson_Correlation', 'Mean_Absolute_Diff', 'Mean_Relative_Diff_Pct']
        final_report_df = final_report_df[cols]
        
        out_csv_path = plots_dir / "stability_report.csv"
        final_report_df.to_csv(out_csv_path, index=False)
        print(f"Successfully processed pairing! Stability evaluated and saved to {out_csv_path}")
    else:
        print("No paired files could be successfully aggregated.")

if __name__ == "__main__":
    main()
