import matplotlib.pyplot as plt
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import itertools

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

csv_path = "/Users/nathan/code/inf1007/data/penguins_size.csv"

penguins_list = []

with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            penguins_list.append({k:v for k,v in row.items()})

## Plot data
x_list = []
y_list = []
s_list = []
result_dict = {}
for key in penguins_list[0].keys():
    result_dict[key] = []

print(penguins_list[0].keys())
for d in penguins_list:
    if d["culmen_length_mm"] != 'NA' and d["culmen_depth_mm"] != 'NA' and d["sex"] != "NA" and d["sex"] != "." :
        for key in penguins_list[0].keys():
            try:
                result_dict[key].append(float(d[key]))
            except:
                result_dict[key].append(d[key])

plot_keys = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Create a figure with subplots
fig = plt.figure(figsize=(12, 12))

for i, (x, y, z) in enumerate(itertools.combinations(plot_keys, 3)):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    ax.view_init(elev=20, azim=30)  # Enable interactive rotation

    class_labels = np.array(result_dict["species"])
    unique_classes = np.unique(class_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))
    
    for cls, color in zip(unique_classes, colors):
        mask = class_labels == cls
        ax.scatter(np.array(result_dict[x])[mask], 
                   np.array(result_dict[y])[mask], 
                   np.array(result_dict[z])[mask], 
                   color=color, 
                   label=cls, 
                   marker='o')

    ax.set_title(f'3D Scatter Plot {i+1}')
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_zlabel(f'{z}')
    ax.legend(title="Classes")

plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.9, 
#                     top=0.9, bottom=0.1, 
#                     wspace=0.4, hspace=0.4)
plt.show()
