import os, csv

def main():
    all_metrics_path = "/Users/nathan/Desktop/all_metrics_tssXtotalXspineps"
    structures = os.listdir(all_metrics_path)
    metrics_list = ['DiceSimilarityCoefficient', 'HausdorffDistance95', 'NormalizedSurfaceDistance']
    methods_list = ['tss', 'total', 'spineps']
    proposed_method = 'tss'
    paired_dict = {}
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
                    input_file = os.path.basename(row['prediction'])
                    if 'input_file' not in metrics_dict[structure][method]:
                        metrics_dict[structure][method]['input_file'] = []
                    metrics_dict[structure][method]['input_file'].append(input_file)
                    for metric in metrics_list:
                        if metric not in metrics_dict[structure][method].keys():
                            metrics_dict[structure][method][metric] = []
                        metrics_dict[structure][method][metric].append(float(row[metric]))

    # Create paired_dict for statistical tests
    for structure in metrics_dict.keys():
        if structure not in paired_dict.keys():
            paired_dict[structure] = {}
        for method in metrics_dict[structure].keys():
            if method == proposed_method:
                continue
            if method not in paired_dict[structure].keys():
                paired_dict[structure][method] = {}
            for metric in metrics_list:
                baseline_values = metrics_dict[structure][method][metric]
                proposed_values = metrics_dict[structure][proposed_method][metric]
                print()
                paired_dict[structure][method][metric] = {
                    'baseline': baseline_values,
                    'proposed': proposed_values
                }



def add_subject_metrics(paired_dict, metrics_dict, row, metrics_list, proposed_method):
    for methods in methods_list:
        if not methods in metrics_dict.keys():
            metrics_dict[methods] = {}
            paired_dict[methods] = {}
        
        for metric in metrics_list:
            if not metric in metrics_dict[methods].keys():
                metrics_dict[methods][metric] = {}
                paired_dict[methods][metric] = {}

            # Remove only failed detections
            if float(row[f"{metric}_{methods}"]) != -1:
                if not contrast in metrics_dict[methods][metric]:
                    metrics_dict[methods][metric][contrast] = []
                if not "all" in metrics_dict[methods][metric]:
                    metrics_dict[methods][metric]["all"] = []
                metrics_dict[methods][metric][contrast].append(float(row[f"{metric}_{methods}"]))
                metrics_dict[methods][metric]["all"].append(float(row[f"{metric}_{methods}"]))

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