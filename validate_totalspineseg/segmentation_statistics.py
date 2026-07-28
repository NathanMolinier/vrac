import os, csv
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def main():
    all_metrics_path = "/Users/nathan/Desktop/all_metrics_tssXtotalXspineps/comparison_instance"
    structures = [s for s in os.listdir(all_metrics_path) if os.path.isdir(os.path.join(all_metrics_path, s)) and len(os.listdir(os.path.join(all_metrics_path, s))) > 0]
    metrics_list = ['DiceSimilarityCoefficient', 'HausdorffDistance95', 'NormalizedSurfaceDistance']
    methods_list = ['tss', 'total', 'spineps']
    proposed_method = 'tss'
    metrics_dict = {}

    for structure in structures:
        structure_path = os.path.join(all_metrics_path, structure)
        if structure not in metrics_dict.keys():
            metrics_dict[structure] = {}
        metrics_files = [f for f in os.listdir(structure_path) if f.endswith('.csv')]
        for metrics_file in metrics_files:
            metrics_file_path = os.path.join(structure_path, metrics_file)
            method = metrics_file.split('_')[1].split('.')[0]

            if method not in metrics_dict[structure].keys():
                metrics_dict[structure][method] = {}

            with open(metrics_file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file = os.path.basename(row['prediction'])
                    label = row['label'].replace(' ','')
                    match_key = f"{file}_{label}" 
                    if 'match_key' not in metrics_dict[structure][method]:
                        metrics_dict[structure][method]['match_key'] = []
                    metrics_dict[structure][method]['match_key'].append(match_key)
                    for metric in metrics_list:
                        if metric not in metrics_dict[structure][method].keys():
                            metrics_dict[structure][method][metric] = []
                        metrics_dict[structure][method][metric].append(float(row[metric]))

    paired_dict = {}
    for structure in structures:
        if structure not in paired_dict.keys():
            paired_dict[structure] = {}
        for method in metrics_dict[structure].keys():
            if method == proposed_method:
                continue
            if method not in paired_dict[structure].keys():
                paired_dict[structure][method] = {}
            for metric in metrics_list:
                if not metric in paired_dict[structure][method].keys():
                    paired_dict[structure][method][metric] = {}
                baseline_values = metrics_dict[structure][method][metric]
                proposed_values = metrics_dict[structure][proposed_method][metric]
                new_baseline_values = []
                new_proposed_values = []
                for idx, match_key in enumerate(metrics_dict[structure][method]['match_key']):
                    if match_key in metrics_dict[structure][proposed_method]['match_key']:
                        proposed_idx = metrics_dict[structure][proposed_method]['match_key'].index(match_key)
                        new_baseline_values.append(baseline_values[idx])
                        new_proposed_values.append(proposed_values[proposed_idx])
                paired_dict[structure][method][metric]["baseline"] = new_baseline_values
                paired_dict[structure][method][metric]["proposed"] = new_proposed_values

    # Compute mean and std
    stats = {}
    for structure in metrics_dict.keys():
        if not structure in stats.keys():
            stats[structure] = {}
        for method in metrics_dict[structure].keys():
            if not method in stats[structure].keys():
                stats[structure][method] = {}
            for metric in metrics_list:
                if not metric in stats[structure][method].keys():
                    stats[structure][method][metric] = {}
                values = metrics_dict[structure][method][metric]
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                stats[structure][method][metric]['mean'] = mean
                stats[structure][method][metric]['std'] = std

                # Confidence interval (95%)
                ci = 1.96 * std / (len(values) ** 0.5)
                stats[structure][method][metric]['ci min'] = mean - ci
                stats[structure][method][metric]['ci max'] = mean + ci

                print(f"\n{structure} - {method} - {metric}")
                print(f"mean={mean:.4f}, std={std:.4f}, ci=({mean - ci:.4f}, {mean + ci:.4f})")

    # Run pairwise Wilcoxon signed-rank tests
    for structure in paired_dict.keys():
        print(f"\nStructure: {structure}")
        for metric in metrics_list:
            raw_p_values = []
            comparisons = []
            for method in paired_dict[structure].keys():
                if method == proposed_method:
                    continue

                baseline_values = paired_dict[structure][method][metric]["baseline"]
                proposed_values = paired_dict[structure][method][metric]["proposed"]

                if len(baseline_values) != len(proposed_values):
                    raise ValueError(f"Baseline and proposed values must have the same length for {structure} - {metric}")
                
                # Perform Wilcoxon signed-rank test
                stat, p_value = wilcoxon(proposed_values, baseline_values, alternative='two-sided')
                raw_p_values.append(p_value)
                comparisons.append(method)
    
            # Apply Benjamini-Hochberg FDR correction for p-values
            reject_null, corrected_p_values, _, _ = multipletests(
                raw_p_values, 
                alpha=0.05, 
                method='fdr_bh'
            ) 
    
            print("-"*50)
            print(" "*25 +f"{metric}" + " "*25)
            print("-"*50)
            for i, name in enumerate(comparisons):
                sig = "*" if reject_null[i] else " "
                print(f"Proposed vs {name}:")
                print(f"  Raw p-value:       {raw_p_values[i]:.4f}")
                print(f"  Corrected p-value: {corrected_p_values[i]:.4f} {sig}")
                print("-" * 50)

if __name__ == "__main__":
    main()