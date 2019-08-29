import os

import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
sys.path.append('/home/jordan/Developement/Thesis/Material Classification/minc-model')
import caffe
import os
import numpy
import glob
import argparse


import numpy as np
import matplotlib.pyplot as plt

def construct_command(N=None, gpu=None, net_prototxt=None, net_weights=None, datadir=None, filelist=None, outfile=None):
    
    local_values = locals()

    cmd = './find_max_acts.py'

    for key, value in local_values.items():
        print(key, value, type(value))
        # if(value != None and value != 'None' and type(value) is str):
        #     cmd += " --{} '{}'".format(str(key), str(value))
        if(key == 'lr_params' or key == 'data_size'):
            cmd += " --{} \"{}\"".format(str(key.replace('_','-')), str(value))
        elif(value != None and value != 'None'):
            cmd += " --{} {}".format(str(key), str(value))

    print(cmd)
    return cmd


parser = argparse.ArgumentParser(description='Finds images in a training set that cause max activation for a network; saves results in a pickled NetMaxTracker.')
parser.add_argument('--N', type = int, default = 9, help = 'note and save top N activations')
parser.add_argument('--gpu', action = 'store_true', help = 'use gpu')
parser.add_argument('net_prototxt', type = str, default = '', help = 'network prototxt to load')
parser.add_argument('net_weights', type = str, default = '', help = 'network weights to load')
parser.add_argument('datadir', type = str, default = '.', help = 'directory to look for files in')
parser.add_argument('filelist', type = str, help = 'list of image files to consider, one per line')
parser.add_argument('outfile', type = str, help = 'output filename for pkl')
#parser.add_argument('--mean', type = str, default = '', help = 'data mean to load')
args = parser.parse_args()

construct_command(args.N, args.gpu, args.net_prototxt, args.net_weights, args.datadir, args.filelist, args.outfile)
