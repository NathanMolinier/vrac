import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    folder_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/metrics_output'
    demographics_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/lbp-lumbar-usf-2025/participants.tsv'

    demographics = pd.read_csv(demographics_path, sep='\t')

    thickness_dict = {}
    intensity_dict = {}
    solidity_dict = {}
    volume_dict = {}
    eccentricity_dict = {}
    img_dict = {}
    seg_dict = {}
    age_dict = {}
    filename_dict = {}
    sex_dict = {}
    filter = ''
    for sub in os.listdir(folder_path):
        csv_folder = os.path.join(folder_path, sub, "csv")
        discs_imgs = os.path.join(folder_path, sub, "imgs")
        sub_name = sub.split('_')[0]
        sub_info = demographics[demographics['participant_id'] == sub_name]
        if os.path.exists(csv_folder) and filter in sub:
            discs_data = pd.read_csv(os.path.join(csv_folder, "discs.csv"))
            vertebrae_data = pd.read_csv(os.path.join(csv_folder, "vertebrae.csv"))

            for name, intensity_peaks_gap, thickness, solidity, eccentricity, volume in zip(discs_data.name, discs_data.intensity_peaks_gap, discs_data.median_thickness, discs_data.solidity, discs_data.eccentricity, discs_data.volume):
                if name not in intensity_dict:
                    intensity_dict[name] = {}
                    thickness_dict[name] = {}
                    solidity_dict[name] = {}
                    eccentricity_dict[name] = {}
                    volume_dict[name] = {}
                    img_dict[name] = {}
                    seg_dict[name] = {}
                    age_dict[name] = {}
                    sex_dict[name] = {}
                    filename_dict[name] = {}

                # Find in dataframe overlying_vert in name
                overlying_vert = name.split('-')[0]
                matching_rows = vertebrae_data[vertebrae_data['name'] == overlying_vert]
                age = sub_info['age'].iloc[0]
                if not matching_rows.empty and (isinstance(age, float) or isinstance(age, int)):
                    if sub_name not in thickness_dict[name].keys():
                        thickness_dict[name][sub_name] = []
                        intensity_dict[name][sub_name] = []
                        solidity_dict[name][sub_name] = []
                        eccentricity_dict[name][sub_name] = []
                        volume_dict[name][sub_name] = []
                        img_dict[name][sub_name] = []
                        seg_dict[name][sub_name] = []
                        age_dict[name][sub_name] = []
                        sex_dict[name][sub_name] = []
                        filename_dict[name][sub_name] = []
                    ap_thickness = float(vertebrae_data[vertebrae_data['name'] == overlying_vert]['AP_thickness'].iloc[0])
                    thickness_dict[name][sub_name].append(thickness/ap_thickness)
                    intensity_dict[name][sub_name].append(intensity_peaks_gap)

                    # Add solidity and eccentricity
                    solidity_dict[name][sub_name].append(solidity)
                    eccentricity_dict[name][sub_name].append(eccentricity)

                    # Add volume normalized by vertebrae volume
                    vert_volume = float(vertebrae_data[vertebrae_data['name'] == overlying_vert]['volume'].iloc[0])
                    volume_dict[name][sub_name].append(volume/vert_volume)

                    # Add image
                    img_dict[name][sub_name].append(np.rot90(plt.imread(os.path.join(discs_imgs, f'discs_{name}_img.png'))))
                    seg_dict[name][sub_name].append(np.rot90(plt.imread(os.path.join(discs_imgs, f'discs_{name}_seg.png'))))

                    # File names
                    filename_dict[name][sub_name].append(sub)

                    # Add age and sex
                    age_dict[name][sub_name].append(age)
                    sex_dict[name][sub_name].append(sub_info['sex'].iloc[0])


    # Normalize thickness_dict values by their median
    for name in thickness_dict:
        thickness_dict = np.array(thickness_dict[name])
        intensity_dict = np.array(intensity_dict[name])
        if len(thickness_array) > 0:
            median_thickness = np.median(thickness_array[intensity_array>0.3]) # median of healthy discs only
            if median_thickness == 0:
                median_thickness = 1.0
                print(f"Warning: median thickness is zero for disc {name}. Setting to 1.0 to avoid division by zero.")
            thickness_dict[name] = thickness_array / median_thickness
        else:
            thickness_dict[name] = thickness_array

    plt.scatter(np.concatenate([thickness_dict[name] for name in thickness_dict]), np.concatenate([intensity_dict[name] for name in intensity_dict]), c=np.concatenate([age_dict[name] for name in age_dict]))
    plt.xlabel('Thickness')
    plt.ylabel('Intensity')
    plt.ylim(0, 2.0)
    plt.xlim(0, 2)
    plt.title('Disc Intensity vs Thickness (All Discs)')
    plt.legend()
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    plt.savefig('imgs/disc_intensity_vs_thickness_by_age.png')


def line(x, a, b):
    return a * x + b

def exp(x, a, b):
    return b + np.exp(a * x)

def convert_str_to_list(string):
    return [float(item) for item in string[1:-1].split(',')]

if __name__ == "__main__":
    main()