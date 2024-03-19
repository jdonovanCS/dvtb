import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
import caffe
import os
import glob

image_dir = '/home/jordan/Development/Thesis/deep-visualization-toolbox/find_maxes/max_imgs'
for directory in os.listdir(image_dir):
    print(directory)
    for subdirectory in os.listdir(directory):
        print(subdirectory)
    # fig = plt.figure(figsize=(224,224))
    # for i in range(0, 9):
    #     img = ()