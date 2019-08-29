import argparse
import numpy as np
import ipdb as pdb
import cPickle as pickle
import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
sys.path.append('/home/jordan/Development/Thesis/MaterialClassification/minc-model')
sys.path.append('/home/jordan/Development/Thesis/deep-visualization-toolbox/')
sys.path.append('/home/jordan/Development/Thesis/deep-visualization-toolbox/find_maxes')
import settings
from loaders import load_imagenet_mean, load_labels, caffe
from jby_misc import WithTimer
from max_tracker import output_max_patches

pickle_in = open('/home/jordan/Development/Thesis/testing_output2', 'rb')
example = pickle.load(pickle_in)

print(example.max_trackers['inception_4b/1x1'].max_vals)

pickle_in = open('/home/jordan/Development/Thesis/deep-visualization-toolbox/optimize_results/test2/opt_inception_5a/5x5_0020_0_info_big.pkl')
example = pickle.load(pickle_in)
print(example[1].best_xx.shape)