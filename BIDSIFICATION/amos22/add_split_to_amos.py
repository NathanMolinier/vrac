import os, csv, json

def main():
    input_folder = '/Users/nathan/data/amos22/amos22'
    out_bids_folder = '/Users/nathan/data/amos22/abdominal-amos22'

    participants_tsv_path = os.path.join(out_bids_folder, 'participants.tsv')
    participants_json_path = os.path.join(out_bids_folder, 'participants.json')

    image_folders = [f for f in os.listdir(input_folder) if f.startswith('images') and os.path.isdir(os.path.join(input_folder, f))]

    split_data = {}
    split_names = {'Tr':'train', 'Va':'val', 'Ts':'test'}
    for folder in image_folders:
        split = split_names[folder.replace('images', '')]
        folder_path = os.path.join(input_folder, folder)
        split_subjects = [name.replace('.nii.gz', '') for name in os.listdir(folder_path) if name.startswith('amos')]
        for subject in split_subjects:  
            split_data[subject] = split

    with open(participants_json_path, 'r') as json_file:
        participants_description = json.load(json_file)

    participants_description['split'] = {"Description": "Dataset split for the amos challenge. Test set do not have segmentations.", "LongName": "Dataset Split", "Levels": {"train": "Training Set", "val": "Validation Set", "test": "Test Set"}}

    with open(participants_json_path, 'w') as json_file:
        json.dump(participants_description, json_file, indent=4)

    participants_dict = {}
    with open(participants_tsv_path, 'r') as tsv_file:
        tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            row['split'] = split_data[row['source_id']]
            participants_dict[row['source_id']] = row

    with open(participants_tsv_path, 'w', newline='') as tsv_file:
        fieldnames = list(participants_dict[next(iter(participants_dict))].keys())
        tsv_writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
        tsv_writer.writeheader()
        for subject_id, row in participants_dict.items():
            tsv_writer.writerow(row)

if __name__ == "__main__":
    main()