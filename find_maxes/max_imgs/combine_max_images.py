import numpy as np
import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
import os
import skimage.io
from PIL import Image


image_dir = '/media/jordan/LACIE/dvtb/find_maxes/max_imgs'
print(filter(lambda x: os.path.isdir(os.path.join(image_dir, x)), os.listdir(image_dir)))
for layer_prefix in filter(lambda x: os.path.isdir(os.path.join(image_dir, x)), os.listdir(image_dir)):
    layer_prefix_path = os.path.join(image_dir, layer_prefix)
    print(layer_prefix)
    print(layer_prefix_path)
    print(filter(lambda a: os.path.isdir(os.path.join(layer_prefix_path, a)), os.listdir(layer_prefix_path)))
    for layer_suffix in filter(lambda y: os.path.isdir(os.path.join(layer_prefix_path,y)), os.listdir(layer_prefix_path)):
        print(layer_suffix)
        layer_suffix_path = os.path.join(layer_prefix_path, layer_suffix)
        print(filter(lambda b: os.path.isdir(os.path.join(layer_suffix_path, b)), os.listdir(layer_suffix_path)))
        for node in filter(lambda z: os.path.isdir(os.path.join(layer_suffix_path,z)), os.listdir(layer_suffix_path)):
            img_list = []
            # try:
            print(node)
            node_path = os.path.join(layer_suffix_path, node)
            for i in range(0, min(9, len(os.listdir(os.path.join(node_path))))):
                img_list.append(os.path.join(node_path, 'maxim_%03d.png' % (i)))
            if len(img_list) < 9:
                continue
            images = map(Image.open, img_list)
            widths, heights = zip(*(j.size for j in images))
            total_width = max([sum(widths[0:3]), sum(widths[4:6]), sum(widths[6:])])
            total_height = max([sum([heights[0],heights[3],heights[6]]), sum([heights[1],heights[4],heights[7]]), sum([heights[2],heights[5],heights[8]])])
            print('total_width: {}, total_height: {}'.format(total_width, total_height))
            new_img = Image.new('RGB', (total_width, total_height))
            x_offset = 0
            y_offset = 0
            img_count = 1
            for im in images:
                print('img_count: {}, x_offset: {}, y_offset: {}'.format(img_count, x_offset, y_offset))
                new_img.paste(im, (x_offset, y_offset))
                
                if img_count % 3 == 0:
                    y_offset += im.size[0]
                    x_offset = 0
                else:
                    x_offset += im.size[0]
                img_count+=1
            new_img.save(os.path.join(node_path, 'max.png'))
            print('saved_image: {}'.format(os.path.join(node_path, 'max.png')))
            # except:
            #     print('ERROR')
            #     continue
if layer_prefix == 'conv1':
    exit()
                
    # fig = plt.figure(figsize=(224,224))
    # for i in range(0, 9):
    #     img = ()