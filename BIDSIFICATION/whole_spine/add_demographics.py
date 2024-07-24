import csv
import os

def main():
    csv_path = '/Users/nathan/Desktop/demographics_ws.csv'
    dataset_path = '/Users/nathan/data/whole-spine'

    # Load demographic file
    with open(csv_path, mode='r') as csvfile:
        spamreader = csv.reader(csvfile)
        demo_dict = {}
        demo_keys = {}
        for i, row in enumerate(spamreader):
            if i == 0:
                for i, key in enumerate(row):
                    if key:
                        demo_keys[key] = i
            else:
                if row[demo_keys['PAM50']] == '1':
                    demo_dict[row[demo_keys['Subject']]] = row[:len(demo_keys.keys())]

    # Load participant.tsv
    with open(os.path.join(dataset_path, 'participants.tsv'), 'r') as tsv_file:
        ws_reader = csv.reader(tsv_file, delimiter='\t')
        ws_dict = {}
        ws_keys = {}
        for i, row in enumerate(ws_reader):
            if i == 0:
                for i, key in enumerate(row):
                    if key:
                        ws_keys[key] = i
            else:
                ws_dict[row[ws_keys['data_id']]] = row[:len(ws_keys.keys())]
    
    match_sub = {} 
    for sub in demo_dict.keys():
        comp_sub = [] # Check compatible subjects and make sure only one is present
        for data_id in ws_dict.keys():
            if (sub in data_id or sub.replace('-', '_') in data_id or sub.split('-')[0] in data_id or sub.replace('p', 'P') in data_id) and demo_dict[sub][demo_keys['Study']] != 'errsm' or f'errsm_{sub}' in data_id:
                comp_sub.append(data_id)
        if len(comp_sub) > 1:
            raise ValueError(f'Multiple compatibility for subject {sub}: {"/n".join(comp_sub)}')
        elif len(comp_sub) == 0:
            print(f'Subject {sub} not present in whole spine dataset')
        else:
            data_id = comp_sub[0]
            if sub == 'ED-20130729' or sub == 'VC-20130726':
                print(f'Excluding subject {sub}')
            else:
                print(f'Matching {sub} with {data_id}')
                if data_id not in match_sub.keys():
                    match_sub[data_id] = sub
                else:
                    # Compare informations
                    assert demo_dict[sub][demo_keys['Center']] == demo_dict[match_sub[data_id]][demo_keys['Center']]
                    assert demo_dict[sub][demo_keys['Age']] == demo_dict[match_sub[data_id]][demo_keys['Age']]
                    assert demo_dict[sub][demo_keys['Height (cm)']] == demo_dict[match_sub[data_id]][demo_keys['Height (cm)']]
                    assert demo_dict[sub][demo_keys['Pathology']] == demo_dict[match_sub[data_id]][demo_keys['Pathology']]
                    assert demo_dict[sub][demo_keys['Sex']] == demo_dict[match_sub[data_id]][demo_keys['Sex']]
                    assert demo_dict[sub][demo_keys['Weight (kg)']] == demo_dict[match_sub[data_id]][demo_keys['Weight (kg)']]
    
    # Add new fields
    new_fields = list(ws_keys.keys()) + ['species', 'age', 'sex', 'height (cm)', 'weight (kg)', 'pathology', 'institution']

    # Add new information
    for ws_id, csv_sub in match_sub.items():
        ws_dict[ws_id]+= [
            'homo sapiens', 
            demo_dict[csv_sub][demo_keys['Age']], 
            demo_dict[csv_sub][demo_keys['Sex']],
            demo_dict[csv_sub][demo_keys['Height (cm)']],
            demo_dict[csv_sub][demo_keys['Weight (kg)']],
            demo_dict[csv_sub][demo_keys['Pathology']], 	
            demo_dict[csv_sub][demo_keys['Center']],
        ]

    # Add n/a for missing values
    for ws_id in ws_dict.keys():
        if len(ws_dict[ws_id]) < len(new_fields):
            ws_dict[ws_id] = ws_dict[ws_id] + ['homo sapiens'] + ['n/a']*(len(new_fields)-len(ws_dict[ws_id]) -2) + [ws_id[:3]]
        for i, val in enumerate(ws_dict[ws_id]):
            if not val or val == '-':
                ws_dict[ws_id][i] = 'n/a'

    # Write tsv file
    participants_tsv_list = [v for v in ws_dict.values()]
    path_out = os.path.join(dataset_path, 'participants_new.tsv')
    with open(path_out, 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(new_fields)
        participants_tsv_list = sorted(participants_tsv_list, key=lambda a : a[0])
        for item in participants_tsv_list:
            tsv_writer.writerow(item)
        print(f'participants.tsv created in {path_out}')


    return

if __name__ == '__main__':
    main()