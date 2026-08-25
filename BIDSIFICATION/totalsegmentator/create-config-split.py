import csv
import glob

def main():
    csv_path = '/Users/nathan/data/fullbody-totalsegmentator-ct/participants.tsv'
    dataset_path = '/Users/nathan/data/fullbody-totalsegmentator-ct'

    # Read CSV file
    participants_dict = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        participants = [row for row in csv_reader]
        participants_dict = {row['participant_id']: row for row in participants}

    # Add patients to split lists
    train, val, test = [], [], []
    for participant_id, row in participants_dict.items():
        image_path = f"{dataset_path}/derivatives/labels/{participant_id}/anat/{participant_id}_CT_label-body_dseg.nii.gz"
        if row['split'] == 'train':
            train.append(image_path)
        elif row['split'] == 'val':
            val.append(image_path)
        elif row['split'] == 'test':
            test.append(image_path)
        else:
            print(f"Warning: Participant {participant_id} has an unknown split value: {row['split']}")
    
    # Create txt files
    with open(f'{dataset_path}/train.txt', 'w') as f:
        for item in train:
            f.write(f"{item}\n")
    with open(f'{dataset_path}/val.txt', 'w') as f:
        for item in val:
            f.write(f"{item}\n")
    with open(f'{dataset_path}/test.txt', 'w') as f:
        for item in test:
            f.write(f"{item}\n")


if __name__ == "__main__":
    main()