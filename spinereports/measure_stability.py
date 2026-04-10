import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

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
        
        # Drop NaNs for valid pairwise comparison
        valid_idx = val_t1.notna() & val_t2.notna()
        v1 = val_t1[valid_idx]
        v2 = val_t2[valid_idx]
        
        if len(v1) == 0:
            continue
            
        abs_diff = np.abs(v2 - v1)
        rel_diff = abs_diff / (np.abs(v1) + 1e-9) # avoid division by zero
        
        corr = v1.corr(v2) if len(v1) > 1 else np.nan
        
        stability_results.append({
            'Metric': col,
            'Mean_Absolute_Diff': abs_diff.mean(),
            'Mean_Relative_Diff_Pct': rel_diff.mean() * 100,
            'Pearson_Correlation': corr,
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
        
        exclude_cols = ['structure', 'structure_name', 'slice_interp', 'vertebra_level', 'subject']
        numeric_cols = [c.replace('_T1w', '') for c in df_all.columns if c.endswith('_T1w') and not any(ext in c for ext in exclude_cols)]
        
        # Calculate stability
        stability_df = calculate_stability(df_all, numeric_cols)
        stability_df['File'] = file_name
        final_reports.append(stability_df)

    if final_reports:
        # 3. Export global structural stability report
        final_report_df = pd.concat(final_reports, ignore_index=True)
        
        # Reorder columns for readability
        cols = ['File', 'Metric', 'N_Samples', 'Pearson_Correlation', 'Mean_Absolute_Diff', 'Mean_Relative_Diff_Pct']
        final_report_df = final_report_df[cols]
        
        final_report_df.to_csv(output_dir, index=False)
        print(f"Successfully processed pairing! Stability evaluated and saved to {output_dir}")
    else:
        print("No paired files could be successfully aggregated.")

if __name__ == "__main__":
    main()
