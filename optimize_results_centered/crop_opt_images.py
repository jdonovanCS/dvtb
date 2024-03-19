import numpy as np
import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
import os
import skimage.io
from PIL import Image
import re
import math

file_size_dict = {
    'conv1/7x7_s2': 7,
    'conv2/3x3_reduce': 19,
    'conv2/3x3': 19,
    'inception_3a/1x1': 27,
    'inception_3a/3x3_reduce': 27,
    'inception_3a/3x3': 43,
    'inception_3a/5x5_reduce': 43,
    'inception_3a/5x5': 75,
    'inception_3a/pool_proj': 91,
    'inception_3b/1x1': 91,
    'inception_3b/3x3_reduce': 91,
    'inception_3b/3x3': 107,
    'inception_3b/5x5_reduce': 107,
    'inception_3b/5x5': 139,
    'inception_3b/pool_proj': 155,
    'inception_4a/1x1': 171,
    'inception_4a/3x3_reduce': 171,
    'inception_4a/3x3': 195,
    'inception_4a/5x5_reduce': 227,
    'inception_4a/5x5': 227,
    'inception_4a/pool_proj': 231,
    'inception_4b/1x1': 231,
    'inception_4b/3x3_reduce': 231,
    'inception_4b/3x3': 231,
    'inception_4b/5x5_reduce': 231,
    'inception_4b/5x5': 231,
    'inception_4b/pool_proj': 231,
    'inception_4c/1x1': 231,
    'inception_4c/3x3_reduce': 231,
    'inception_4c/3x3': 231,
    'inception_4c/5x5_reduce': 231,
    'inception_4c/5x5': 231,
    'inception_4c/pool_proj': 231,
    'inception_4d/1x1': 231,
    'inception_4d/3x3_reduce': 231,
    'inception_4d/3x3': 231,
    'inception_4d/5x5_reduce': 231,
    'inception_4d/5x5': 231,
    'inception_4d/pool_proj': 231,
    'inception_4e/1x1': 231,
    'inception_4e/3x3_reduce': 231,
    'inception_4e/3x3': 231,
    'inception_4e/5x5_reduce': 231,
    'inception_4e/5x5': 231,
    'inception_4e/pool_proj': 231,
    'inception_5b/1x1': 231,
    'inception_5b/3x3_reduce': 231,
    'inception_5b/3x3': 231,
    'inception_5b/5x5_reduce': 231,
    'inception_5b/5x5': 231,
    'inception_5b/pool_proj': 231,
    'inception_5a/1x1': 231,
    'inception_5a/3x3_reduce': 231,
    'inception_5a/3x3': 231,
    'inception_5a/5x5_reduce': 231,
    'inception_5a/5x5': 231,
    'inception_5a/pool_proj': 231}

image_dir = '/media/jordan/LACIE/dvtb/optimize_results_centered/'
test_dir = 'test4'
image_dir = os.path.join(image_dir, test_dir)
save_dir = '/media/jordan/LACIE/dvtb/optimize_results_centered/test4/cropped'
for layer_prefix in filter(lambda x: os.path.isdir(os.path.join(image_dir, x)), os.listdir(image_dir)):
    layer_prefix_path = os.path.join(image_dir, layer_prefix)
    # if 'conv1' not in layer_prefix:
    #     continue
    print(layer_prefix)
    print(layer_prefix_path)
    print(filter(lambda a: os.path.isdir(os.path.join(layer_prefix_path, a)), os.listdir(layer_prefix_path)))
    for image in filter(lambda y: 'X.jpg' in os.path.join(layer_prefix_path, y), os.listdir(layer_prefix_path)):
        print(image)
        im = Image.open(os.path.join(layer_prefix_path, image))
        width, height = im.size

        precise_layer = re.sub('opt_', '', layer_prefix) + '/' + re.sub('_\d\d\d\d_0_best_X.jpg', '', image)
        print(precise_layer)
        if precise_layer not in file_size_dict.keys():
            continue
        # precise_layer = layer_prefix + '/' + re.sub('_%d%d%d%d_0_beset_X.jpg', '', image)

        left = (width / 2) - math.floor(file_size_dict[precise_layer] / 2.0)
        right = (width / 2) + math.ceil(file_size_dict[precise_layer] / 2.0)
        top = (height/2) - math.floor(file_size_dict[precise_layer] / 2.0)
        bottom = (height/2) + math.ceil(file_size_dict[precise_layer]/2.0)

        # left = (width / 2) - max(file_size_dict[precise_layer] - 112, 0)
        # right = (width / 2) + max(file_size_dict[precise_layer], 112)
        # top = (height / 2) - max(file_size_dict[precise_layer] - 112, 0)
        # bottom = (height / 2) + max(file_size_dict[precise_layer], 112)

        print(left, right, top, bottom)
        print(os.path.join(save_dir, layer_prefix))
        if not os.path.isdir(os.path.join(save_dir, layer_prefix)):
            os.mkdir(os.path.join(save_dir, layer_prefix))#, re.sub('/[0-9]*[A-Za-z]*[0-9]*[_]*[\d]*[_]*[\d]*[_]*[A-Za-z]*[_]*[A-Za-z]*[.]*[A-Za-z]*', '/', image)))
        new_im = im.crop((left, top, right, bottom))
        new_im.save(os.path.join(save_dir, layer_prefix, image))
        