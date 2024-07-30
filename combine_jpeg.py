import os
from PIL import Image
import cv2
import numpy as np

jpeg_folder = '/Users/nathan/Desktop/preview_freeze_decoder'
sub_string = 'step1'

nb_column = 6
col_list = []
line_list = []
img_name = []
jpeg_list = [jpeg for jpeg in os.listdir(jpeg_folder) if '.jpg' in jpeg and sub_string in jpeg]

# Get max image shape for padding
shape = []
for i, jpeg in enumerate(jpeg_list):
    if sub_string in jpeg:
        jpeg_path = os.path.join(jpeg_folder, jpeg)
        im = np.array(Image.open(jpeg_path))
        shape.append(im.shape)
shape = np.array(shape)

# Combine images after padding along rows and columns
for i, jpeg in enumerate(jpeg_list):
    if sub_string in jpeg:
        jpeg_path = os.path.join(jpeg_folder, jpeg)
        im = np.array(Image.open(jpeg_path))
        x_width = np.max(shape[:, 0]) - im.shape[0]
        y_width = np.max(shape[:, 1]) - im.shape[1]
        im = np.pad(im, pad_width=((x_width//2,x_width-x_width//2), (y_width//2,y_width-y_width//2), (0,0)))

        col_list.append(im)
        img_name.append(jpeg)

        if len(col_list) == nb_column:
            line_list.append(np.concatenate(col_list, axis=1))
            col_list = []
        
        if i == len(jpeg_list)-1:
            last_row = np.concatenate(col_list, axis=1)
            extra_pad = line_list[0].shape[1] - last_row.shape[1]
            last_row = np.pad(last_row, pad_width=((0,0), (extra_pad//2,extra_pad-extra_pad//2), (0,0)))
            line_list.append(last_row)
    else:
        print(f'Not considering {jpeg}')

out_img = np.concatenate(line_list, axis=0)
cv2.imwrite(os.path.join(jpeg_folder,f'res_{sub_string}.png'), out_img)
