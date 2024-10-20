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

def title2image(title, width, height):
    """
    :param title: String title
    :param width: width in pixels

    :return: title as image
    """
    # Create title
    title = np.where(np.array(text_to_image(title))[:,:,0]==255,255,0)
    
    # Keep title aspect ratio
    ratio = (title.shape[0])/height
    orig_shape_title = title.shape
    title = Image.fromarray(title[:,:].astype(np.uint8))
    if round(orig_shape_title[1]/ratio)<=width:
        title = np.array(title.resize((round(orig_shape_title[1]/ratio), round(orig_shape_title[0]/ratio)), Image.LANCZOS))
    else:
        ratio = width/height
        title = np.array(title.resize((round(orig_shape_title[1]/ratio), round(orig_shape_title[0]/ratio)), Image.LANCZOS))
    # Repeat title along the third dimension
    title = np.repeat(title[:,:,np.newaxis],3,axis=2)

    # Pad title
    width_pad = width - title.shape[1]
    title = np.pad(title, pad_width=((0,0), (width_pad//2, width_pad - width_pad//2), (0,0)))

    return title

def main():
    jpeg_folder = ['/Users/nathan/Desktop/preview/loc', '/Users/nathan/Desktop/preview/raw', '/Users/nathan/Desktop/preview/raw_loc']
    titles = ['A', 'B', 'C']
    sub_string = ''

    col_list = []
    line_list = []
    img_name = []
    jpeg_list = [jpeg for jpeg in os.listdir(jpeg_folder[0]) if '.jpg' in jpeg and sub_string in jpeg]
    txt_height = 60

    # Extract optimal number of line and column
    power = np.ceil(np.log(len(jpeg_list))/np.log(2))
    nb_line = round(2**(power//2))
    nb_col = round(2**(power - power//2))
    assert nb_col*nb_line >= len(jpeg_list)
    
    # Extract shape
    shape = []
    for i, jpeg in enumerate(jpeg_list):
        if sub_string in jpeg:
            for folder in jpeg_folder:
                jpeg_folder_path = os.path.join(folder, jpeg)
                im = np.array(Image.open(jpeg_folder_path))
                shape.append(im.shape)
    shape = np.array(shape)

    # Combine images after padding along rows and columns
    for i, jpeg in enumerate(jpeg_list):
        if sub_string in jpeg:
            img_list = []
            shape_list = []
            for title, folder in zip(titles, jpeg_folder):
                jpeg_folder_path = os.path.join(folder, jpeg)
                im = np.array(Image.open(jpeg_folder_path))

                # Pad image
                x_width = np.max(shape[:,0])-im.shape[0]
                gap = 5
                im = np.pad(im, pad_width=((x_width//2+gap,x_width - x_width//2),(0,0),(0,0)), mode='constant')

                # Create title
                title_img = title2image(title=title, width=im.shape[1], height=txt_height)

                # Concatenate image and title
                im = np.concatenate((title_img, im), axis=0)
                img_list.append(im)
                shape_list.append(im.shape)

            # Merge images + Add short padding to the right
            im = np.concatenate([np.pad(im, pad_width=((0,0),(0,20),(0,0)), mode='constant') if i+1 < len(img_list) else im for i, im in enumerate(img_list)], axis=1)

            # Create title
            contrast = jpeg.split('_sag')[0].split('_')[-1]
            title_contrast = title2image(title=contrast, width=im.shape[1], height=txt_height)

            # Concat title with images
            im = np.concatenate((im, title_contrast), axis=0)
            
            # Add padding to the side to distinguish images
            im = np.pad(im, pad_width=((10,10),(10,10),(0,0)), mode='constant')
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
    cv2.imwrite(os.path.join(jpeg_folder[0],f'validate_crop_{sub_string}.png'), out_img)

if __name__=='__main__':
    main()
