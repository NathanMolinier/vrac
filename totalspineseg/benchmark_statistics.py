import csv

def main():
    # Compute benchmark statistics for totalspineseg article
    metrics_folder = "/Users/nathan/Desktop/20251210_results"
    contrasts = ["T1w", "T2w"]
    metrics_dict = {}
    metrics_list = ['l2_mean', 'Accuracy']
    methods_list = ['hourglass_T1w_T2w', 'sct', 'totalspineseg', 'spinenet', 'tss_c7-t1', 'tss_t12-l1', 'tss_c2-c3', 'tss_all']

    # Fecth metrics from csv
    for contrast in contrasts:
        file = f"computed_metrics_{contrast}.csv"
        metrics_path = f"{metrics_folder}/{file}"
        # Load and process the metrics
        with open(metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['subject'].startswith('sub-'):
                    add_subject_metrics(metrics_dict, contrast, row, metrics_list, methods_list)

    # Compute mean and std
    stats = {}
    for methods in metrics_dict.keys():
        if not methods in stats.keys():
            stats[methods] = {}
        for metric in metrics_dict[methods].keys():
            if not metric in stats[methods].keys():
                stats[methods][metric] = {}
            for contrast in metrics_dict[methods][metric].keys():
                if not contrast in stats[methods][metric]:
                    stats[methods][metric][contrast] = {}
                values = metrics_dict[methods][metric][contrast]
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                stats[methods][metric][contrast]['mean'] = mean
                stats[methods][metric][contrast]['std'] = std

                # Confidence interval (95%)
                ci = 1.96 * std / (len(values) ** 0.5)
                stats[methods][metric][contrast]['ci min'] = mean - ci
                stats[methods][metric][contrast]['ci max'] = mean + ci

                if contrast == "all":
                    print(f"{methods} - {metric} - {contrast}: mean={mean:.4f}, std={std:.4f}, ci=({mean - ci:.4f}, {mean + ci:.4f})")

    # Compute statistics between methods
    
            
def add_subject_metrics(metrics_dict, contrast, row, metrics_list, methods_list):
    for methods in methods_list:
        if not methods in metrics_dict.keys():
            metrics_dict[methods] = {}
        
        for metric in metrics_list:
            if not metric in metrics_dict[methods].keys():
                metrics_dict[methods][metric] = {}
            # Remove failed detections
            if float(row[f"{metric}_{methods}"]) != -1:
                if not contrast in metrics_dict[methods][metric]:
                    metrics_dict[methods][metric][contrast] = []
                if not "all" in metrics_dict[methods][metric]:
                    metrics_dict[methods][metric]["all"] = []
                metrics_dict[methods][metric][contrast].append(float(row[f"{metric}_{methods}"]))
                metrics_dict[methods][metric]["all"].append(float(row[f"{metric}_{methods}"]))
            



if __name__ == "__main__":
    main()