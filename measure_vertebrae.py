import os, json
import numpy as np
from progress.bar import Bar

from vrac.data_management.image import Image
from vrac.plot.plot import save_boxplot

# Vert dict
VERT_DICT = {
    "C1":1,
    "C2":2,
    "C3":3,
    "C4":4,
    "C5":5,
    "C6":6,
    "C7":7,
    "T1":8,
    "T2":9,
    "T3":10,
    "T4":11,
    "T5":12,
    "T6":13,
    "T7":14,
    "T8":15,
    "T9":16,
    "T10":17,
    "T11":18,
    "T12":19,
    "T13":28,
    "L1":20,
    "L2":21,
    "L3":22,
    "L4":23,
    "L5":24,
    "L6":25,
    "sacrum":[26, 29, 30, 31, 32],
    "cocygis":28
}

def main():
    # Path to config
    config_path = '/home/GRAMES.POLYMTL.CA/p118739/data/config_data/vert-seg/verse-spine-ct.json'
    
    # Read json file and create a dictionary
    with open(config_path, "r") as file:
        config = json.load(file)

    plot_dict = {}
    for split in ['TRAINING', 'VALIDATION', 'TESTING']:
        dict_list = config[split]
        bar = Bar(f'Load {split} data', max=len(dict_list))
        for d in dict_list:
            # Get segmentation path
            seg_path = os.path.join(config['DATASETS_PATH'], d['LABEL'])

            # Load segmentation
            seg = Image(seg_path).change_orientation('RSP')

            # Get image resolutions
            Rres, Sres, Pres = seg.dim[4:7]

            # Uniq vert
            unique_vert = np.unique(seg.data)

            for vert, value in VERT_DICT.items():
                if any(np.isin(unique_vert, value)):
                    # Create mask of the vertebra
                    if not isinstance(value, list):
                        non_zeros = np.where(seg.data == value)
                    else:
                        non_zeros = np.where(np.isin(seg.data, value))

                    # Get indices max and min indices
                    Rmin, Rmax = np.min(non_zeros[0]), np.max(non_zeros[0])
                    Smin, Smax = np.min(non_zeros[1]), np.max(non_zeros[1])
                    Pmin, Pmax = np.min(non_zeros[2]), np.max(non_zeros[2])

                    # Get vertebrae size using resolution
                    Rsize = (Rmax - Rmin)*Rres
                    Ssize = (Smax - Smin)*Sres
                    Psize = (Pmax - Pmin)*Pres

                    # Add vert to plot dict
                    if vert not in plot_dict.keys():
                        plot_dict[vert] = {"R":[Rsize], "S":[Ssize], "P":[Psize]}
                    else:
                        plot_dict[vert]["R"].append(Rsize)
                        plot_dict[vert]["S"].append(Ssize)
                        plot_dict[vert]["P"].append(Psize)
            # Plot progress
            bar.suffix  = f'{dict_list.index(d)+1}/{len(dict_list)}'
            bar.next()
        bar.finish()
    # Plot boxplot
    save_boxplot(plot_dict.keys(), [plot_dict[vert]["R"] for vert in plot_dict.keys()], output_path='vert_sizeRL.png', x_axis='Vertebrae size along the Right-Left direction', y_axis='Size (mm)')
    save_boxplot(plot_dict.keys(), [plot_dict[vert]["S"] for vert in plot_dict.keys()], output_path='vert_sizeSI.png', x_axis='Vertebrae size along the Superior-Inferior direction', y_axis='Size (mm)')
    save_boxplot(plot_dict.keys(), [plot_dict[vert]["P"] for vert in plot_dict.keys()], output_path='vert_sizePA.png', x_axis='Vertebrae size along the Posterior-Anterior direction', y_axis='Size (mm)')
    print()

                
                        

        



if __name__=='__main__':
    main()