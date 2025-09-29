import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def main():
    folder_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/metrics_output'
    
    thickness_dict = {}
    intensity_dict = {}
    img_dict = {}
    seg_dict = {}
    for sub in os.listdir(folder_path):
        csv_folder = os.path.join(folder_path, sub, "csv")
        discs_imgs = os.path.join(folder_path, sub, "imgs")
        if os.path.exists(csv_folder):
            discs_data = pd.read_csv(os.path.join(csv_folder, "discs.csv"))
            vertebrae_data = pd.read_csv(os.path.join(csv_folder, "vertebrae.csv"))
            for name, intensity_peaks_gap, thickness in zip(discs_data.name, discs_data.intensity_peaks_gap, discs_data.median_thickness):
                if name not in intensity_dict:
                    intensity_dict[name] = []
                    thickness_dict[name] = []
                    img_dict[name] = []
                    seg_dict[name] = []
                    gt_dict[name] = []

                # Find in dataframe overlying_vert in name
                overlying_vert = name.split('-')[0]
                matching_rows = vertebrae_data[vertebrae_data['name'] == overlying_vert]
                if not matching_rows.empty:
                    ap_thickness = float(vertebrae_data[vertebrae_data['name'] == overlying_vert]['AP_thickness'].iloc[0])
                    thickness_dict[name].append(thickness/ap_thickness)
                    intensity_dict[name].append(intensity_peaks_gap)

                    # Add image
                    img_dict[name].append(np.rot90(plt.imread(os.path.join(discs_imgs, f'discs_{name}_img.png'))))
                    seg_dict[name].append(np.rot90(plt.imread(os.path.join(discs_imgs, f'discs_{name}_seg.png'))))
    # Generate subplots
    # for name in intensity_dict:
    #     # Fit a curve to the data
    #     plt.figure()
    #     x = np.array(thickness_dict[name])
    #     y = np.array(intensity_dict[name])
    #     plt.plot(x, y, 'o', label=name)
    #     plt.ylabel('Intensity')
    #     plt.title(f'Disc: {name}')
    #     plt.legend()
    #     plt.xlabel('Thickness')
    #     plt.tight_layout()
    #     plt.savefig(f'imgs/disc_intensity_vs_thickness_subplots_{name}.png')
    #     plt.close()

    # Normalize thickness_dict values by their median
    for name in thickness_dict:
        thickness_array = np.array(thickness_dict[name])
        intensity_array = np.array(intensity_dict[name])
        if len(thickness_array) > 0:
            median_thickness = np.median(thickness_array[intensity_array>0.8])
            thickness_dict[name] = thickness_array / median_thickness
        else:
            thickness_dict[name] = thickness_array
    
    # Determine discs grades
    grades_dict = {}
    for name in intensity_dict:
        if name not in grades_dict:
            grades_dict[name] = []
        thickness_array = np.array(thickness_dict[name])
        intensity_array = np.array(intensity_dict[name])
        for thickness, intensity in zip(thickness_array, intensity_array):
            if thickness < 0.3 and intensity < 0.2:
                grades_dict[name].append(8)
            elif thickness < 0.6 and intensity < 0.2:
                grades_dict[name].append(7)
            elif thickness < 0.9 and intensity < 0.2:
                grades_dict[name].append(6)
            elif intensity < 0.1 and thickness >= 0.9:
                grades_dict[name].append(5)
            elif intensity < 0.3 and thickness >= 0.9:
                grades_dict[name].append(4)
            elif intensity < 0.6 and thickness >= 0.9:
                grades_dict[name].append(3)
            elif intensity < 0.9 and thickness >= 0.9:
                grades_dict[name].append(2)
            elif intensity >= 0.9 and thickness >= 0.9:
                grades_dict[name].append(1)
            else:
                grades_dict[name].append(0) # error
    # x_line = np.linspace(-1, 1, 200)
    # grades = {}
    # for i in range(0, 8):
    #     grades[8-i] = {'a': -3, 'b': -1.5 + 0.60 * i} # line
    #     # grades[8-i] = {'a': -1, 'b': -1.8 + 0.5 * i} # exp
    # grades_dict = {}
    # for name in intensity_dict:
    #     grades_dict[name] = []
    #     for x, y in zip(thickness_dict[name], intensity_dict[name]):
    #         graded = False
    #         grade = 8
    #         while not graded:
    #             if y < line(x, grades[grade]['a'], grades[grade]['b']):
    #                 graded = True
    #             else:
    #                 if grade == 1:
    #                     grade = 0
    #                     graded = True
    #                 else:
    #                     grade -= 1
    #         grades_dict[name].append(grade)

    # Plot general plot with all points
    plt.figure(figsize=(10, 5))
    # Regroup all discs in plot
    # for name in thickness_dict:
    #     plt.scatter(thickness_dict[name], intensity_dict[name], label=name)
    # Plot all points together with color corresponding to grade
    plt.scatter(np.concatenate([thickness_dict[name] for name in thickness_dict]), np.concatenate([intensity_dict[name] for name in intensity_dict]), c=np.concatenate([grades_dict[name] for name in grades_dict]))
    plt.xlabel('Thickness')
    plt.ylabel('Intensity')
    plt.ylim(0, 2.0)
    plt.xlim(0, 2)
    plt.title('Disc Intensity vs Thickness (All Discs)')
    plt.legend()
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    plt.savefig('imgs/disc_intensity_vs_thickness.png')

    # Create subplots for each disc with rows corresponding to grades and pick 5 examples per grade if possible
    for name in intensity_dict:
        grades_list = np.array(grades_dict[name])
        imgs = img_dict[name]
        segs = seg_dict[name]
        thicknesses = np.array(thickness_dict[name])
        intensities = np.array(intensity_dict[name])
        # Combine image and segmentation side by side for each example
        combined_imgs = [np.concatenate((img, seg), axis=1) for img, seg in zip(imgs, segs)]
        imgs = combined_imgs
        unique_grades = np.unique(grades_list)
        if grades_list.size > 0:
            n_grades = len(unique_grades)
            n_examples = 5

            fig, axes = plt.subplots(n_grades, n_examples, figsize=(n_examples * 3, n_grades * 3))
            if n_grades == 1:
                axes = np.expand_dims(axes, 0)
            for i, grade in enumerate(unique_grades):
                idxs = np.where(grades_list == grade)[0]
                if len(idxs) > n_examples:
                    idxs = np.random.choice(idxs, n_examples, replace=False)
                for j in range(n_examples):
                    ax = axes[i, j] if n_grades > 1 else axes[0, j]
                    ax.axis('off')
                    if j < len(idxs):
                        img = imgs[idxs[j]]
                        thickness_val = thicknesses[idxs[j]]
                        intensity_val = intensities[idxs[j]]
                        ax.imshow(img, cmap='gray')
                        if grade == 0:
                            title = f'Grade error'
                        else:
                            title = f'Grade {grade}'
                        # Display thickness and intensity values
                        ax.set_title(f'{title}\nT={thickness_val:.2f}, I={intensity_val:.2f}', fontsize=16)
                    else:
                        ax.set_visible(False)
            plt.suptitle(f'Examples for disc {name}', fontsize=32)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'imgs/disc_{name}_examples_by_grade.png')
            plt.close()

def line(x, a, b):
    return a * x + b

def exp(x, a, b):
    return b + np.exp(a * x)

def convert_str_to_list(string):
    return [float(item) for item in string[1:-1].split(',')]

if __name__ == "__main__":
    main()