import matplotlib.pyplot as plt
import cv2
from vrac.utils.utils import normalize
import numpy as np

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
    