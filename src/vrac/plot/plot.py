import matplotlib.pyplot as plt
import cv2
from vrac.utils.utils import normalize
import numpy as np
import pandas as pd
import seaborn as sns

def save_bar(names, values, output_path, x_axis, y_axis):
    '''
    Create a histogram plot
    :param names: String list of the names
    :param values: Values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: y-axis name

    '''
            
    # Set position of bar on X axis
    fig = plt.figure(figsize = (len(names)//2, 5))
 
    # creating the bar plot
    plt.bar(names, values, width = 0.4)
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(names)
    plt.title("Discs distribution")
    plt.savefig(output_path)

def save_features(features, path_out):
    '''
    Save image feature maps
    :param features: Individual features
    :param path_out: Save path
    '''
    # Detach features from GPU
    features = features.detach().cpu().numpy()[0]

    # Extract shape
    shape = features.shape

    # Extract middle sagittal slice
    features = features[:, shape[1]//2, :, :]

    if shape[0] > 1:
        # Extract optimal number of line and column
        power = np.log(shape[0])/np.log(2)
        nb_line = round(2**(power//2))
        nb_col = round(2**(power - power//2))
        assert nb_col*nb_line >= shape[0]
        
        # Create column and line list
        col_list = []
        line_list = []
        for i in range(shape[0]):
            feature = features[i]
            col_list.append(normalize(feature)*255)

            if len(col_list) == nb_col:
                line_list.append(np.concatenate(col_list, axis=1))
                col_list = []
            
            if i == shape[0] - 1 and col_list:
                last_row = np.concatenate(col_list, axis=1)
                extra_pad = line_list[0].shape[1] - last_row.shape[1]
                last_row = np.pad(last_row, pad_width=((0,0), (extra_pad//2,extra_pad-extra_pad//2)))
                line_list.append(last_row)
    
        # Concatenate lines
        out_img = np.concatenate(line_list, axis=0)

    else:
        out_img = normalize(features[0])*255

    # Save image
    cv2.imwrite(path_out, out_img)


def save_nested_pie(output_path, inner_labels, outer_labels, inner_sizes, outer_sizes, inner_colors=None, outer_colors=None):
    """
    Create a nested pie chart with two levels.

    Parameters:
        inner_labels (list): Labels for the inner pie.
        outer_labels (list): Labels for the outer pie.
        inner_sizes (list): Sizes for the inner pie.
        outer_sizes (list): Sizes for the outer pie.
        inner_colors (list, optional): Colors for the inner pie.
        outer_colors (list, optional): Colors for the outer pie.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

    # Generate random colors if none are provided
    if inner_colors is None:
        inner_colors = plt.cm.get_cmap('Set2', len(inner_labels))(np.linspace(0, 1, len(inner_labels))) #["#%06x" % random.randint(0, 0xFFFFFF) for _ in inner_labels]
    if outer_colors is None:
        outer_colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(outer_labels))) #["#%06x" % random.randint(0, 0xFFFFFF) for _ in outer_labels]

    # Create the inner pie chart
    inner_pie, _ = ax.pie(
        inner_sizes,
        labels=None,  # Disable default labels
        colors=inner_colors,
        radius=0.7,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    # Create the outer pie chart
    outer_pie, _ = ax.pie(
        outer_sizes,
        labels=None,  # Disable default labels
        colors=outer_colors,
        radius=1.0,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    # Add label boxes for the inner pie
    for i, (wedge, label, size) in enumerate(zip(inner_pie, inner_labels, inner_sizes)):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = 0.5 * np.cos(np.radians(angle))
        y = 0.5 * np.sin(np.radians(angle))
        ax.text(x, y, label + f" = {size}", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor=inner_colors[i], edgecolor='black', alpha=0.7))

    # Add label boxes for the outer pie
    for i, (wedge, label, size) in enumerate(zip(outer_pie, outer_labels, outer_sizes)):
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = 1.2 * np.cos(np.radians(angle))  # Adjusted distance for better visibility
        y = 1.2 * np.sin(np.radians(angle))
        ax.text(x, y, label + f" = {size}", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor=outer_colors[i], edgecolor='black', alpha=0.7))
    
    plt.savefig(output_path)

def save_pie(output_path, labels, sizes, colors=None):
    """
    Create a nested pie chart with two levels.

    Parameters:
        labels (list): Labels for the pie.
        sizes (list): Sizes for the pie.
        colors (list, optional): Colors for the pie.

    Returns:
        Savefig
    """
    fig, ax = plt.subplots()
    #ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    fig_size = 30
    fig.set_figheight(fig_size)
    fig.set_figwidth(round(1.4*fig_size)) 
    # Generate random colors if none are provided
    if colors is None:
        colors = plt.cm.get_cmap('Set1', len(labels))(np.linspace(0, 1, len(labels))) #["#%06x" % random.randint(0, 0xFFFFFF) for _ in inner_labels]
    
    # Create the inner pie chart
    wedges, _ = ax.pie(
        sizes,
        labels=None,  # Disable default labels
        colors=colors,
        radius=0.7,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    # Add labels outside the chart with connecting lines
    for i, (wedge, label, size) in enumerate(zip(wedges, labels, sizes)):
        if len(labels) == 1:
            angle = 90
        else:
            angle = (wedge.theta1 + wedge.theta2) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))

        # New label
        new_label = f"{label} (n = {size})"
        
        # Label position outside the pie
        if len(new_label) < 15:
            label_x = 1.3 * np.cos(np.radians(angle))
            label_y = 1 * np.sin(np.radians(angle))
        else:
            label_x = 1.5 * np.cos(np.radians(angle))
            label_y = 1.2 * np.sin(np.radians(angle))

        # Line start and end points
        line_x = [x*0.7, label_x]
        line_y = [y*0.7, label_y]

        ax.plot(line_x, line_y, color='black', linewidth=0.3*fig_size)  # Connecting line
        ax.text(
            label_x, label_y, new_label, ha='center', va='center', fontweight='bold', size=fig_size*2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black', alpha=1)
        )
    
    plt.savefig(output_path, transparent=True)


def save_boxplot(methods, values, hue=[], output_path='test.png', x_axis='Methods', y_axis='Metric name'):
    '''
    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark/blob/main/src/bcm/utils/plot.py

    Create a boxplot graph
    :param methods: String list of the methods name
    :param values: Values associated with the methods
    :param hue: Class associated with the methods
    :param output_path: Path to output folder where figures will be stored
    :param x_axis: x-axis name
    :param y_axis: y-axis name
    '''
    # set width of bar
    # Set position of bar on X axis
    result_dict = {'methods' : [], 'values' : [], 'Class' : []}
    for i, method in enumerate(methods):
        if len(values[i]) > 0:
            result_dict['values'] += values[i]
            for j in range(len(values[i])):
                result_dict['methods'] += [method]
                result_dict['Class'] += [hue[i]] if hue else [method]


    result_df = pd.DataFrame(data=result_dict)
    sns.set_theme(style="darkgrid")

    # Make the plot 
    plt.figure(figsize=(max(len(methods), 13), 10))
    ax = sns.boxplot(x="methods", y="values", hue="Class", data=result_df, width=0.9)
    if len(methods) > 20:
        xticks = [f'{method}' if i%2==0 else f'\n{method}' for i, method in enumerate(methods)] # Shift label up and down
        plt.xticks(list(range(len(xticks))), xticks, fontsize=14)
    else:
        plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} violin plot')
    plt.xlabel(x_axis, fontsize = 20)
    plt.ylabel(y_axis, fontsize = 20)
    plt.title(y_axis, fontsize = 25)

    # Extract the colors used for the hue from the plot
    handles, labels = ax.get_legend_handles_labels()
    legend_colors = [handle.get_facecolor() for handle in handles]

    if legend_colors:
        # Match the x-tick labels' colors to the corresponding hue colors
        for i, label in enumerate(ax.get_xticklabels()):
            method = methods[i]  # Corresponding method
            # Retrieve the associated class for the current method
            associated_class = result_df[result_df["methods"] == method]["Class"].values[0]
            class_idx = list(result_df["Class"].unique()).index(associated_class)  # Find the index of the class
            label.set_color(legend_colors[class_idx])  # Set the color based on the hue class

        # Increase the font size of the legend (hue label)
        plt.legend(title='Class', title_fontsize=18, fontsize=18, loc='best')

    # Adjust spacing between the plot and labels
    plt.tight_layout()
    
    # Save plot
    print(f'Figure saved under {output_path}')
    plt.savefig(output_path)