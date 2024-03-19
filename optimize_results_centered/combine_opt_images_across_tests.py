import numpy as np
import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
import os
import skimage.io
from PIL import Image
import math


image_dir = '/media/jordan/LACIE/dvtb/optimize_results_centered/'
test_dirs = ['test1', 'test2', 'test3', 'test4']
layers = ['conv2/3x3', 'inception_3a/5x5', 'inception_4a/5x5', 'inception_5a/5x5']


for layer in layers:
    for image in (os.listdir(os.path.join(image_dir, test_dirs[0], 'cropped', 'opt_' + layer[:-4])) + 
        [x for x in os.listdir(os.path.join(image_dir, test_dirs[0], 'cropped')) if 'X.jpg' in x]):
        if ('X.jpg' in image):
            imglist = []
            for test_dir in test_dirs:
                if 'fc8-20' in image and 'conv' in layer:
                    imglist.append(os.path.join(image_dir, test_dir, 'cropped', image))
                elif 'fc8-20' in image:
                    continue
                else:
                    imglist.append(os.path.join(image_dir, test_dir, 'cropped', 'opt_' + layer[:-4], image))
        if len(imglist) < 1:
            continue
        images = map(Image.open, imglist)
        widths, heights = zip(*(j.size for j in images))
        total_width = int(widths[0] * math.ceil(math.sqrt(len(widths))))
        total_height = int(heights[0] * math.ceil(math.sqrt(len(heights))))
        new_img = Image.new('RGB', (total_width, total_height))
        x_offset = 0
        y_offset = 0
        img_count = 1
        for im in images:
            print('img_count: {}, x_offset: {}, y_offset: {}'.format(img_count, x_offset, y_offset))
            new_img.paste(im, (x_offset, y_offset))
            
            if img_count % (math.ceil(math.sqrt(len(images)))) == 0:
                y_offset += im.size[0]
                x_offset = 0
            else:
                x_offset += im.size[0]
            img_count+=1
        if 'fc8-20' in image:
            new_img.save(os.path.join(image_dir, 'combined_filters_across_tests_' + image[:-4] + '.png'))
        else:
            new_img.save(os.path.join(image_dir, 'combined_filters_across_tests_' + layer[:-4] + '_' + image[:-4] + '.png'))
        print('saved_image: {}'.format(os.path.join(image_dir, 'combined_filters_across_tests_' + layer[:-4] + '_' + image[:-4] + '.png')))


# if layer_prefix == 'conv1':
#     exit()