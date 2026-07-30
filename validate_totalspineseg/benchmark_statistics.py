import csv
import numpy as np
from scipy.stats import wilcoxon, bootstrap
from statsmodels.stats.multitest import multipletests

def main():
    # Compute benchmark statistics for totalspineseg article
    metrics_folder = "/Users/nathan/Desktop/files/20251210_results"
    contrasts = ["T1w", "T2w"]
    metrics_dict = {}
    paired_dict = {}
    metrics_list = ['l2_mean', 'Accuracy']
    methods_list = ['hourglass_T1w_T2w', 'sct', 'spinenet', 'tss_c7-t1', 'tss_t12-l1', 'tss_c2-c3', 'tss_all']
    proposed_method = 'tss_all'

    # Fetch metrics from csv
    for contrast in contrasts:
        file = f"computed_metrics_{contrast}.csv"
        metrics_path = f"{metrics_folder}/{file}"
        # Load and process the metrics
        with open(metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['subject'].startswith('sub-'):
                    add_subject_metrics(paired_dict, metrics_dict, contrast, row, metrics_list, methods_list, proposed_method)

    # Compute mean and std
    stats = {}
    for methods in metrics_dict.keys():
        if not methods in stats.keys():
            stats[methods] = {}
        for metric in metrics_dict[methods].keys():
            if not metric in stats[methods].keys():
                stats[methods][metric] = {}
            for contrast in metrics_dict[methods][metric].keys():
                if contrast == "failed":
                    continue
                if not contrast in stats[methods][metric]:
                    stats[methods][metric][contrast] = {}
                values = metrics_dict[methods][metric][contrast]
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                stats[methods][metric][contrast]['mean'] = mean
                stats[methods][metric][contrast]['std'] = std

                # Bootstrap confidence interval (95%)
                scipy_bootstrap = bootstrap(
                    (np.array(values),), 
                    np.mean, 
                    confidence_level=0.95, 
                    n_resamples=10000, 
                    method='bca',
                    random_state=42
                )

                scipy_ci_low = scipy_bootstrap.confidence_interval.low
                scipy_ci_high = scipy_bootstrap.confidence_interval.high

                stats[methods][metric][contrast]['ci min'] = scipy_ci_low
                stats[methods][metric][contrast]['ci max'] = scipy_ci_high

                if contrast == "all":
                    print(f"\n{methods} - {metric}")
                    print(f"mean={mean:.4f}, std={std:.4f}, ci=({scipy_ci_low:.4f}, {scipy_ci_high:.4f})")
                    print(f"failed={metrics_dict[methods][metric]['failed']}")

    # Run pairwise Wilcoxon signed-rank tests
    for metric in metrics_list:
        raw_p_values = []
        comparisons = []
        for method in paired_dict.keys():
            if method.startswith('tss_all'):
                continue
        
            baseline_values = np.array(paired_dict[method][metric]["baseline"])
            proposed_values = np.array(paired_dict[method][metric]["proposed"])

            if len(baseline_values) != len(proposed_values):
                raise ValueError(f"Baseline and proposed values must have the same length for {methods} - {metric}")

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

            
def add_subject_metrics(paired_dict, metrics_dict, contrast, row, metrics_list, methods_list, proposed_method):
    for methods in methods_list:
        if not methods in metrics_dict.keys():
            metrics_dict[methods] = {}
            paired_dict[methods] = {}
        
        for metric in metrics_list:
            if not metric in metrics_dict[methods].keys():
                metrics_dict[methods][metric] = {}
                paired_dict[methods][metric] = {}
                metrics_dict[methods][metric]["failed"] = 0

            # Remove only failed detections
            if float(row[f"{metric}_{methods}"]) != -1:
                if not contrast in metrics_dict[methods][metric]:
                    metrics_dict[methods][metric][contrast] = []
                if not "all" in metrics_dict[methods][metric]:
                    metrics_dict[methods][metric]["all"] = []
                metrics_dict[methods][metric][contrast].append(float(row[f"{metric}_{methods}"]))
                metrics_dict[methods][metric]["all"].append(float(row[f"{metric}_{methods}"]))
            else:                    
                metrics_dict[methods][metric]["failed"] += 1

            # Add paired values
            if float(row[f"{metric}_{methods}"]) != -1 and float(row[f"{metric}_{proposed_method}"]) != -1:
                if not "baseline" in paired_dict[methods][metric]:
                    paired_dict[methods][metric]["baseline"] = []
                if not "proposed" in paired_dict[methods][metric]:
                    paired_dict[methods][metric]["proposed"] = []
                paired_dict[methods][metric]["baseline"].append(float(row[f"{metric}_{methods}"]))
                paired_dict[methods][metric]["proposed"].append(float(row[f"{metric}_{proposed_method}"]))

            



if __name__ == "__main__":
    main()