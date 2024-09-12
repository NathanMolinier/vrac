import os
from PIL import Image, ImageFont, ImageDraw, ImageColor
import cv2
import numpy as np

def text_to_image(
    text: str,
    font_filepath='/Library/Fonts/arialuni.ttf',
    font_size= 70,
    color=(255,255,255)):
    # From https://stackoverflow.com/questions/68648801/generate-image-from-given-text
    
    font = ImageFont.truetype(font_filepath, size=font_size)

    size = font.getmask(text, 'L').size
    img = Image.new("RGBA", (size[0],size[1]*2))

    draw = ImageDraw.Draw(img)
    draw_point = (0, 0)

    draw.multiline_text(draw_point, text, font=font, fill=color)

    text_window = img.getbbox()
    img = img.crop(text_window)

    return img

jpeg_folder_raw = '/Users/nathan/Desktop/preview_raw'
jpeg_folder_crop = '/Users/nathan/Desktop/preview_cropreg'
sub_string = '_step2_output'

col_list = []
line_list = []
img_name = []
jpeg_list = [jpeg for jpeg in os.listdir(jpeg_folder_raw) if '.jpg' in jpeg and sub_string in jpeg]

# Extract optimal number of line and column
power = np.ceil(np.log(len(jpeg_list))/np.log(2))
nb_line = round(2**(power//2))
nb_col = round(2**(power - power//2))
assert nb_col*nb_line >= len(jpeg_list)

# Combine images after padding along rows and columns
for i, jpeg in enumerate(jpeg_list):
    if sub_string in jpeg:
        jpeg_path_raw = os.path.join(jpeg_folder_raw, jpeg)
        jpeg_path_crop = os.path.join(jpeg_folder_crop, jpeg.replace(sub_string,''))
        im_raw = np.array(Image.open(jpeg_path_raw))
        im_crop = np.array(Image.open(jpeg_path_crop))

        # Add short padding to the right
        im_raw = np.pad(im_raw, pad_width=((0,0),(0,20),(0,0)), mode='constant')

        # Merge raw and crop version
        im = np.concatenate((im_raw, im_crop), axis=1)

        # Create title
        title = np.where(np.array(text_to_image(jpeg))[:,:,0]==255,255,0)
        
        # Keep title aspect ratio
        gap = 50
        ratio = (title.shape[1]+gap)/im.shape[1]
        orig_shape_title = title.shape
        title = Image.fromarray(title[:,:].astype(np.uint8))
        title = np.array(title.resize((round(orig_shape_title[1]//ratio), round(orig_shape_title[0]//ratio)), Image.LANCZOS))
        
        # Repeat title along the third dimension
        title = np.repeat(title[:,:,np.newaxis],3,axis=2)

        # Pad title
        width_pad = im.shape[1] - title.shape[1]
        title = np.pad(title, pad_width=((0,0), (width_pad//2, im.shape[1] - title.shape[1] - width_pad//2), (0,0)))

        # Concat title with images
        im = np.concatenate((im, title), axis=0)

        # Add padding to the side to distinguish images
        im = np.pad(im, pad_width=((5,5),(5,5),(0,0)), mode='constant')
        im = np.pad(im, pad_width=((5,5),(5,5),(0,0)), mode='constant', constant_values=255)
        im = np.pad(im, pad_width=((5,5),(5,5),(0,0)), mode='constant')

        col_list.append(im)

        if i == len(jpeg_list)-1:
            # Pad images to have same height
            new_col = []
            comb_vert_max = 0
            for comb in col_list:
                if comb.shape[0] > comb_vert_max:
                    comb_vert_max = comb.shape[0]
            for comb in col_list:
                height = comb_vert_max - comb.shape[0]
                new_col.append(np.pad(comb, pad_width=((height//2,height - height//2), (0,0), (0,0))))
            line_list.append(np.concatenate(new_col, axis=1))

            # Pad image to have same width
            new_line = []
            comb_horiz_max = 0
            for comb in line_list:
                if comb.shape[1] > comb_horiz_max:
                    comb_horiz_max = comb.shape[1]
            for comb in line_list:
                width = comb_horiz_max - comb.shape[1]
                new_line.append(np.pad(comb, pad_width=((0,0), (width//2,width - width//2), (0,0))))
        
        elif len(col_list) == nb_col:
            # Pad image to have same height
            new_col = []
            comb_vert_max = 0
            for comb in col_list:
                if comb.shape[0] > comb_vert_max:
                    comb_vert_max = comb.shape[0]
            for comb in col_list:
                height = comb_vert_max - comb.shape[0]
                new_col.append(np.pad(comb, pad_width=((height//2,height - height//2), (0,0), (0,0))))
            line_list.append(np.concatenate(new_col, axis=1))
            col_list = []
    else:
        print(f'Not considering {jpeg}')

out_img = np.concatenate(new_line, axis=0)
cv2.imwrite(os.path.join(jpeg_folder_crop,f'validate_crop_{sub_string}.png'), out_img)
