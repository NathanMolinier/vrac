import json
import os
import pandas as pd
import numpy as np

from vrac.data_management.image import Image
from vrac.data_management.utils import fetch_subject_and_session, fetch_contrast

def main():
    #path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/sc-seg/sci-zurich-colorado_dcm-oklahoma.json'
    #path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/canal-seg/dcm-oklahoma-brno_sci-paris.json'
    #path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/vert-labeling/benchmark.json'
    path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/vert-labeling/sexy_data.json'

    # Load json data
    with open(path_json_config, 'r') as file:
        config = json.load(file)

    processed_files = []

    # Init out dict
    total_nb = 0
    resolution=[]
    missing_tsv=[]
    tsv_dict = {
        'age':[],
        'pathology':{},
        'sex':{}
    }
    file_dict = {
        'contrast':{},
        'acquisition':{},
        'dataset':{}
    }

    for dic in config['TESTING']:
        path_img = os.path.join(config['DATASETS_PATH'], dic['IMAGE'])

        # Fetch information from filename
        subjectID, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(path_img)
        mri_contrast = fetch_contrast(path_img)

        # Check if subject already processed
        if filename not in processed_files:
            total_nb+=1
            # Add subject
            processed_files.append(filename)

            # Load image
            img = Image(path_img).change_orientation('RSP')

            # Extract resolution
            dim = img.dim
            resolution.append(list(dim[4:7]))

            # Add file info
            if mri_contrast not in file_dict['contrast'].keys():
                file_dict['contrast'][mri_contrast]=1
            else:
                file_dict['contrast'][mri_contrast]+=1
            
            if acquisition not in file_dict['acquisition'].keys():
                file_dict['acquisition'][acquisition]=1
            else:
                file_dict['acquisition'][acquisition]+=1
            
            dataset =  dic['IMAGE'].split('/')[0]
            if dataset not in file_dict['dataset'].keys():
                file_dict['dataset'][dataset]=1
            else:
                file_dict['dataset'][dataset]+=1

            # Extract participant.tsv information
            # Load participant.tsv information
            path_participant_tsv = os.path.join(config['DATASETS_PATH'], dic['IMAGE'].split('/')[0], 'participants.tsv')
            tsv = pd.read_csv(path_participant_tsv, sep='\t').to_dict()
            participant_idx_dict = {v:k for k,v in tsv['participant_id'].items()}
            if subjectID in participant_idx_dict.keys():
                participant_idx = participant_idx_dict[subjectID]
                for key in tsv_dict.keys():
                    if key in tsv.keys():
                        info = tsv[key][participant_idx]
                    else:
                        if key == 'age':
                            info = np.nan
                        else:
                            info = 'None'
                    if key == 'age':
                        tsv_dict[key].append(info)
                    else:
                        if info not in tsv_dict[key].keys():
                            tsv_dict[key][info]=1
                        else:
                            tsv_dict[key][info]+=1
            else:
                missing_tsv.append(subjectID)


    
    # Print information
    print('--- Age ---')
    print()
    print(f'Mean = {np.nanmean(tsv_dict["age"])}')
    print(f'STD = {np.nanstd(tsv_dict["age"])}')
    print()
    print('--- Pathologies ---')
    print()
    for pat, num in tsv_dict['pathology'].items():
        print(f'{pat} --> N = {num}')
    print()
    print('--- Sex ---')
    print()
    for sex, num in tsv_dict['sex'].items():
        if isinstance(sex, str):
            print(f'{sex} --> N = {num}')
    print()
    print('--- MRI contrasts ---')
    print()
    for info, num in file_dict['contrast'].items():
        print(f'{info} --> N = {num}')
    print()
    print('--- MRI acquisition ---')
    print()
    for info, num in file_dict['acquisition'].items():
        print(f'{info} --> N = {num}')
    print()
    print('--- Datasets ---')
    print()
    for info, num in file_dict['dataset'].items():
        print(f'{info} --> N = {num}')
    print()
    print('--- Total ---')
    print()
    print(f'N = {total_nb}')
    print()










if __name__=='__main__':
    main()