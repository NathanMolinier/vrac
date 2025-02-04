import json
import os
import pandas as pd
import numpy as np
import math

from vrac.data_management.image import Image
from vrac.data_management.utils import fetch_subject_and_session, fetch_contrast
from vrac.plot.plot import save_nested_pie, save_pie

def main():
    #path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/sc-seg/sci-zurich-colorado_dcm-oklahoma.json' # SC seg
    #path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/canal-seg/dcm-oklahoma-brno_sci-paris.json' # canal seg
    #path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/vert-labeling/benchmark.json' # labeling seg
    path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/config/sexy_data.json' # Sexy
    #path_json_config = '/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/config/splits.json' # Train

    # Load json data
    with open(path_json_config, 'r') as file:
        config = json.load(file)

    processed_files = []
    get_resolution = False

    split = 'TESTING'

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
        'dataset':{},
        'dataset_sub':{},
        'subject':[],
        'contxpath':{}
    }

    for dic in config[split]:
        path_img = os.path.join(config['DATASETS_PATH'], dic['IMAGE'])
        dataset = dic['IMAGE'].split('/')[0]

        if '' in dataset:
            # Fetch information from filename
            subjectID, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(path_img)
            mri_contrast = fetch_contrast(path_img)

            # Add subjects
            if subjectID not in file_dict['subject']:
                file_dict['subject'].append(subjectID)

                dataset_sub =  dic['IMAGE'].split('/')[0]
                if dataset_sub not in file_dict['dataset_sub'].keys():
                    file_dict['dataset_sub'][dataset_sub]=1
                else:
                    file_dict['dataset_sub'][dataset_sub]+=1

            # Check if subject already processed
            if filename not in processed_files:
                total_nb+=1
                # Add subject
                processed_files.append(filename)

                if get_resolution:
                    # Load image
                    img = Image(path_img).change_orientation('RPI')

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
                            if key == 'pathology':
                                if info == 'None' or not isinstance(info, str):
                                    if 'sci' in dataset:
                                        info = "SCI"
                                    elif 'dcm' in dataset:
                                        info = "DCM"
                                    elif 'pediatric' in dataset or 'hc-leipzig-7t-mp2rage' in dataset or 'data-single-subject' in dataset or 'whole-spine' in dataset:
                                        info = 'HC'
                                    elif 'canproco' in dataset or 'marseille-3t-mp2rage' in dataset or 'basel-mp2rage' in dataset:
                                        info = 'MS'
                                    elif 'spider-challenge-2023' in dataset:
                                        info = 'LBP'
                                    else:
                                        raise ValueError(f'Missing pathology in {dataset}')
                                contxpath = f"{mri_contrast}x{info}"
                                if contxpath not in file_dict['contxpath'].keys():
                                    file_dict["contxpath"][contxpath]=1
                                else:
                                    file_dict["contxpath"][contxpath]+=1
                            if info not in tsv_dict[key].keys():
                                tsv_dict[key][info]=1
                            else:
                                tsv_dict[key][info]+=1
                else:
                    missing_tsv.append(subjectID)

    # Replace pathology name
    new_pathology = []
    for i, pathology in enumerate(tsv_dict['pathology'].keys()):
        if "MildCompression" in pathology:
            pathology = "MildCompression"
        elif "symptomatic" in pathology:
            pathology = "LBP"
        new_pathology.append(pathology)
    # Save plots
    # Pathology
    save_pie("pathology_pie.png", 
            labels=new_pathology,
            sizes=list(tsv_dict['pathology'].values()),
            )
    # Contrast
    save_pie("contrast_pie.png", 
            labels=list(file_dict['contrast'].keys()),
            sizes=list(file_dict['contrast'].values()),
            )
    
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
    print('--- Contrast X Pathologie ---')
    print()
    for pat, num in file_dict['contxpath'].items():
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
    print('--- Datasets scans ---')
    print()
    for info, num in file_dict['dataset'].items():
        print(f'{info} --> N = {num}')
    print()
    print('--- Datasets patients ---')
    print()
    for info, num in file_dict['dataset_sub'].items():
        print(f'{info} --> N = {num}')
    print()
    print('--- Total subjects ---')
    print()
    print(f'N = {len(file_dict["subject"])}')
    print()
    print('--- Total images ---')
    print()
    print(f'N = {total_nb}')
    print()










if __name__=='__main__':
    main()