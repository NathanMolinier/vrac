import os
from PIL import Image, ImageFont, ImageDraw, ImageColor
import cv2
import numpy as np
import glob

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
    orig_shape_title = title.shape
    ratio = (orig_shape_title[0])/height
    title = Image.fromarray(title[:,:].astype(np.uint8))
    reduce_height = height
    while round(orig_shape_title[1]/ratio)>width:
        reduce_height -= 1
        ratio = (orig_shape_title[0])/reduce_height
    title = np.array(title.resize((round(orig_shape_title[1]/ratio), round(orig_shape_title[0]/ratio)), Image.LANCZOS))
    
    # Repeat title along the third dimension
    title = np.repeat(title[:,:,np.newaxis],3,axis=2)

    # Pad title
    width_pad = width - title.shape[1]
    title = np.pad(title, pad_width=((0,0), (width_pad//2, width_pad - width_pad//2), (0,0)))

    return title

def main():
    jpeg_folder = '/Users/nathan/Desktop/preview'
    sub_string = 'step2_output_tags'

    title_height = 60

    # Fetch jpeg list
    jpeg_list = glob.glob(os.path.join(jpeg_folder, f'*{sub_string}.jpg'))

    # Regroup contrasts
    cont_dict = {}
    for file in jpeg_list:
        contrast = file.split('_sag')[0].split('_')[-1]
        if contrast == 'MTS':
            if 'mt-on' in file:
                contrast = 'MT on'
            else:
                contrast = 'MT off'
        if contrast == 'UNIT1':
            contrast = 'MP2RAGE-UNI'
        if contrast == 'T2star':
            contrast = 'T2*w'
        if 'ct' in file:
            contrast = 'CT'
        if not contrast in cont_dict.keys():
            cont_dict[contrast] = [file]
        else:
            cont_dict[contrast].append(file)

    # Construct contrasts associations
    contrast_dict = {}
    final_shape_dict = {}
    for cont in cont_dict.keys():
        img_list = []
        shape = []
        paths = []
        for jpeg_path in cont_dict[cont]:
            im = np.array(Image.open(jpeg_path))
            img_list.append(im)
            shape.append(im.shape) # Extract shapes
            paths.append(jpeg_path)
        shape = np.array(shape)
        if cont not in ['T1w','T2w']:
            col_list = []
            for im in img_list:
                x_width = np.max(shape[:, 0]) - im.shape[0]
                y_width = np.max(shape[:, 1]) - im.shape[1]
                im = np.pad(im, pad_width=((x_width//2,x_width-x_width//2), (y_width//2,y_width-y_width//2), (0,0)))
                col_list.append(im)

            # Concatenate contrasts into a row
            row = np.concatenate(col_list, axis=1)

            # Create tile
            title = title2image(cont, width=row.shape[1], height=title_height)
            title = np.pad(title, pad_width=((0,5), (0,0), (0,0)), mode='constant')

            # Concat title with images
            row = np.pad(row, pad_width=((0,5), (0,0), (0,0)), mode='constant')
            im = np.concatenate((row, title), axis=0)

            # Pad with colored rectangle
            color = list(np.random.choice(range(256), size=3))
            pad_value = (5,5)
            new_im = np.zeros((im.shape[0]+2*pad_value[0],im.shape[1]+2*pad_value[1],3))
            new_im[:,:,0] = np.pad(im[:,:,0], pad_width=(pad_value, pad_value), mode='constant', constant_values=color[0])
            new_im[:,:,1] = np.pad(im[:,:,1], pad_width=(pad_value, pad_value), mode='constant', constant_values=color[1])
            new_im[:,:,2] = np.pad(im[:,:,2], pad_width=(pad_value, pad_value), mode='constant', constant_values=color[2])
            im = new_im

            # Add small black padding
            im = np.pad(im, pad_width=(pad_value, pad_value, (0,0)), mode='constant')

            # Add contrast row
            #cv2.imwrite('test.png', im)
            contrast_dict[cont]=im
            final_shape_dict[cont]=im.shape
            
        else:
            big_list = []
            small_list = []
            for im in img_list:
                if im.shape[0] == np.max(shape[:, 0]):
                    big_list.append(im)
                else: 
                    y_width = np.max(shape[:, 1]) - im.shape[1]
                    im = np.pad(im, pad_width=((2,2), (y_width//2+2,y_width-y_width//2+2), (0,0)))
                    small_list.append(im)
            # Extract width of all the small images
            x_width_small_list = np.sum([im.shape[0] for im in small_list])

            # Extract all the padding size
            x_width = np.max(shape[:, 0]) - x_width_small_list

            if x_width >= 0:
                # Divide this padding between all the images
                inter_pad = x_width // (len(small_list)+1)

                # Add padding after each image
                small_list = [np.pad(im, pad_width=((0,inter_pad), (0,0), (0,0))) for im in small_list]

                # Add top padding
                top_pad = x_width-len(small_list)*inter_pad
                small_list[0] = np.pad(small_list[0], pad_width=((top_pad,0), (0,0), (0,0)))

                # Concatenate small images
                small_concat = np.concatenate(small_list, axis=0)

                big_img=big_list[0]

            else:
                small_concat = np.concatenate(small_list, axis=0)

                # Pad big image
                x_width = -x_width
                big_img = np.pad(big_list[0], pad_width=((x_width//2,x_width-x_width//2), (0,0), (0,0)))
            
            # Concatenate with big image
            if cont == 'T1w':
                cont_fig = np.concatenate((big_img, small_concat), axis=1)
            else:
                cont_fig = np.concatenate((small_concat, big_img), axis=1)

            # Create tile
            title = title2image(cont, width=cont_fig.shape[1], height=title_height)
            title = np.pad(title, pad_width=((0,5), (0,0), (0,0)), mode='constant')

            # Concat title with images
            cont_fig = np.pad(cont_fig, pad_width=((0,5), (0,0), (0,0)), mode='constant')
            im = np.concatenate((cont_fig, title), axis=0)

            # Pad with colored rectangle
            color = list(np.random.choice(range(256), size=3))
            pad_value = (5,5)
            new_im = np.zeros((im.shape[0]+2*pad_value[0],im.shape[1]+2*pad_value[1],im.shape[2]))
            new_im[:,:,0] = np.pad(im[:,:,0], pad_width=(pad_value, pad_value), mode='constant', constant_values=color[0])
            new_im[:,:,1] = np.pad(im[:,:,1], pad_width=(pad_value, pad_value), mode='constant', constant_values=color[1])
            new_im[:,:,2] = np.pad(im[:,:,2], pad_width=(pad_value, pad_value), mode='constant', constant_values=color[2])
            im = new_im

            # Add small black padding
            im = np.pad(im, pad_width=(pad_value, pad_value, (0,0)), mode='constant')

            # Add contrast row
            #cv2.imwrite('test.png', im)
            contrast_dict[cont]=im
            final_shape_dict[cont]=im.shape
    
    # Merge figures
    row_list = []
    edge_dict = {}
    final_shapes = np.array(list(final_shape_dict.values()))
    for cont, cont_fig in contrast_dict.items():
        if cont not in ['T1w', 'T2w', 'T2star', 'MP2RAGE']:
            y_width = np.max(final_shapes[:, 1]) - cont_fig.shape[1]
            cont_fig = np.pad(cont_fig, pad_width=((0,0), (y_width//2,y_width-y_width//2), (0,0)))
            row_list.append(cont_fig)

    # Concatenate rows
    row_fig = np.concatenate(row_list, axis=0)

    # Pad edge contrasts
    for cont in ['T1w', 'T2w']:
        # Concatenate MP2RAGE with T2w and T2star with T1w
        im = contrast_dict[cont]
        im2 = contrast_dict['T2star'] if cont=='T1w' else contrast_dict['MP2RAGE']
        if im.shape[1] > im2.shape[1]:
            y_width = im.shape[1] - im2.shape[1]
            im2 = np.pad(im2, pad_width=((0,0), (y_width//2,y_width-y_width//2), (0,0)))
        else:
            y_width = im2.shape[1] - im.shape[1]
            im = np.pad(im, pad_width=((0,0), (y_width//2,y_width-y_width//2), (0,0)))
        edge_dict[cont]=np.concatenate((im2,im), axis=0)

    # Identify max x
    x_max = np.max([arr.shape[0] for arr in edge_dict.values()]+[row_fig.shape[0]])
    if row_fig.shape[0] < x_max:
        x_width = x_max - row_fig.shape[0]
        row_fig = np.pad(row_fig, pad_width=((x_width//2,x_width-x_width//2), (0,0), (0,0)))
    final_fig = row_fig

    for cont,im in edge_dict.items():
        if im.shape[0] < x_max:
            x_width = x_max - im.shape[0]
            im = np.pad(im, pad_width=((x_width//2,x_width-x_width//2), (0,0), (0,0)))
        if cont == 'T1w':
            final_fig = np.concatenate((im, final_fig), axis=1)
        else:
            final_fig = np.concatenate((final_fig, im), axis=1)

    cv2.imwrite(os.path.join(jpeg_folder,f'sexy_{sub_string}.png'), final_fig)

if __name__=='__main__':
    main()