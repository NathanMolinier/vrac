import matplotlib.pyplot as plt
import csv
import numpy as np
import seaborn as sns
import pandas as pd

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
fig, ax = plt.subplots(len(plot_keys), len(plot_keys))
# fig.set_figheight(100)
# fig.set_figwidth(100)

result_df = pd.DataFrame(data=result_dict)
sns.set_theme(style="darkgrid")

for i, key1 in enumerate(plot_keys):
     for j, key2 in enumerate(plot_keys):
        gfg = sns.scatterplot(data=result_df, x=key1, y=key2, hue="island", s=10, ax=ax[i,j])
        if i != j:
            gfg.legend_.remove()
        ax[i,j].set_xlabel(key1)
        ax[i,j].set_ylabel(key2)

plt.subplots_adjust(left=0.1, right=0.9, 
                    top=0.9, bottom=0.1, 
                    wspace=0.4, hspace=0.4)
plt.show()
