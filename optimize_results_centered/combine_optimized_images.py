import numpy as np
import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
import os
import skimage.io
from PIL import Image
import math


image_dir = '/media/jordan/LACIE/dvtb/optimize_results_centered/test2/cropped'
print(filter(lambda x: os.path.isdir(os.path.join(image_dir, x)), os.listdir(image_dir)))
for layer_prefix in filter(lambda x: os.path.isdir(os.path.join(image_dir, x)), os.listdir(image_dir)):
    layer_prefix_path = os.path.join(image_dir, layer_prefix)
    print(layer_prefix)
    print(layer_prefix_path)
    print(filter(lambda a: os.path.isfile(os.path.join(layer_prefix_path, a)), os.listdir(layer_prefix_path)))
    img_list = []
    for node in filter(lambda z: os.path.isfile(os.path.join(layer_prefix_path,z)), os.listdir(layer_prefix_path)):
        # try:
        node_path = os.path.join(layer_prefix_path, node)
        if (('_best_X.jpg' in node_path) and ('3x3' in node_path) and ('reduce' not in node_path)) or '7x7' in node_path:
            img_list.append(node_path)
            print(node)
    if (len(img_list) < 1):
        continue
    images = map(Image.open, img_list)
    widths, heights = zip(*(j.size for j in images))
    print('width: {} and sqrt of images: {}'.format(widths[0], math.ceil(math.sqrt(len(widths)))))
    total_width = int(widths[0] * math.ceil(math.sqrt(len(widths)))) # max([sum(widths[0:3]), sum(widths[4:6]), sum(widths[6:])])
    total_height = int(heights[0] * math.ceil(math.sqrt(len(heights)))) # max([sum([heights[0],heights[3],heights[6]]), sum([heights[1],heights[4],heights[7]]), sum([heights[2],heights[5],heights[8]])])
    print('total_width: {}, total_height: {}'.format(total_width, total_height))
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
    new_img.save(os.path.join(layer_prefix_path, 'combined_filters.png'))
    print('saved_image: {}'.format(os.path.join(layer_prefix_path, 'combined_filters.png')))
    # except:
    #     print('ERROR')
    #     continue
if layer_prefix == 'conv1':
    exit()
                
    # fig = plt.figure(figsize=(224,224))
    # for i in range(0, 9):
    #     img = ()