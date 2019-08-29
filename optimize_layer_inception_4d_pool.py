import os

import sys
sys.path.append('/home/jordan/Development/Thesis/caffe/python/')
sys.path.append('/home/jordan/Developement/Thesis/MaterialClassification/minc-model')
import caffe
import os
import numpy
import glob
import argparse


import numpy as np
import matplotlib.pyplot as plt

LR_POLICY_CHOICES = ('constant', 'progress', 'progress01')



def construct_command(caffe_root=None, deploy_proto=None, net_weights=None, mean=None, 
                        channel_swap_to_RGB=None, data_size=None, start_at=None, 
                        rand_seed=None, push_layer=None, push_channel=None, push_spatial=None,
                        push_dir=None, decay=None,
                        blur_radius=None, blur_every=None, small_val_percentile=None, small_norm_percentile=None,
                        px_benefit_percentile=None, px_abs_benefit_percentile=None,
                        lr_policy=None, lr_params=None, max_iter=None, output_prefix=None,
                        output_template=None, brave=None, skip_big=None):
    local_values = locals()
    
    cmd = './optimize_image.py'
    
    for key, value in local_values.items():
        print(key, value, type(value))
        # if(value != None and value != 'None' and type(value) is str):
        #     cmd += " --{} '{}'".format(str(key), str(value))
        if(key == 'lr_params' or key == 'data_size' or key == 'push_spatial'):
            cmd += " --{} \"{}\"".format(str(key.replace('_','-')), str(value))
        elif(key == 'brave'):
            cmd += " --brave"
        elif(value != None and value != 'None'):
            cmd += " --{} {}".format(str(key.replace('_','-')), str(value))

    print(cmd)
    return cmd


parser = argparse.ArgumentParser(description='Script to find, with or without regularization, images that cause high or low activations of specific neurons in a network via numerical optimization. Settings are read from settings.py, overridden in settings_local.py, and may be further overridden on the command line.',
formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100)
)

# Network and data options
# parser.add_argument('--net-weights', type = str, default = settings.caffevis_network_weights,
# help = 'Path to caffe network weights.')
# parser.add_argument('--mean', type = str, default = repr(settings.caffevis_data_mean),
# help = '''Mean. The mean may be None, a tuple of one mean value per channel, or a string specifying the path to a mean image to load. Because of the multiple datatypes supported, this argument must be specified as a string that evaluates to a valid Python object. For example: "None", "(10,20,30)", and "'mean.npy'" are all valid values. Note that to specify a string path to a mean file, it must be passed with quotes, which usually entails passing it with double quotes in the shell! Alternately, just provide the mean in settings_local.py.''')
# parser.add_argument('--channel-swap-to-rgb', type = str, default = '(2,1,0)',
# help = 'Permutation to apply to channels to change to RGB space for plotting. Hint: (0,1,2) if your network is trained for RGB, (2,1,0) if it is trained for BGR.')
parser.add_argument('--data-size', type = str, default='(227, 227)',
# default = '(227,227)',
help = 'Size of network input.')

#### FindParams

# Where to start
# parser.add_argument('--start-at', type = str, default = 'mean_plus_rand', choices = ('mean_plus_rand', 'randu', 'mean'),
# help = 'How to generate x0, the initial point used in optimization.')
parser.add_argument('--rand-seed', type = int, default = 0,
help = 'Random seed used for generating the start-at image (use different seeds to generate different images).')

# What to optimize
# parser.add_argument('--push-layer', type = str, default = 'fc8',
# help = 'Name of layer that contains the desired neuron whose value is optimized.')
# parser.add_argument('--push-channel', type = int, default = '130',
# help = 'Channel number for desired neuron whose value is optimized (channel for conv, neuron index for FC).')
parser.add_argument('--push-spatial', type = str, default = 'None',
help = 'Which spatial location to push for conv layers. For FC layers, set this to None. For conv layers, set it to a tuple, e.g. when using `--push-layer conv5` on AlexNet, --push-spatial (6,6) will maximize the center unit of the 13x13 spatial grid.')
parser.add_argument('--push-dir', type = float, default = 1,
help = 'Which direction to push the activation of the selected neuron, that is, the value used to begin backprop. For example, use 1 to maximize the selected neuron activation and  -1 to minimize it.')

# Use regularization?
parser.add_argument('--decay', type = float, default = 0,
help = 'Amount of L2 decay to use.')
parser.add_argument('--blur-radius', type = float, default = 0,
help = 'Radius in pixels of blur to apply after each BLUR_EVERY steps. If 0, perform no blurring. Blur sizes between 0 and 0.3 work poorly.')
parser.add_argument('--blur-every', type = int, default = 0,
help = 'Blur every BLUR_EVERY steps. If 0, perform no blurring.')
parser.add_argument('--small-val-percentile', type = float, default = 0,
help = 'Induce sparsity by setting pixels with absolute value under SMALL_VAL_PERCENTILE percentile to 0. Not discussed in paper. 0 to disable.')
parser.add_argument('--small-norm-percentile', type = float, default = 0,
help = 'Induce sparsity by setting pixels with norm under SMALL_NORM_PERCENTILE percentile to 0. \\theta_{n_pct} from the paper. 0 to disable.')
parser.add_argument('--px-benefit-percentile', type = float, default = 0,
help = 'Induce sparsity by setting pixels with contribution under PX_BENEFIT_PERCENTILE percentile to 0. Mentioned briefly in paper but not used. 0 to disable.')
parser.add_argument('--px-abs-benefit-percentile', type = float, default = 0,
help = 'Induce sparsity by setting pixels with contribution under PX_BENEFIT_PERCENTILE percentile to 0. \\theta_{c_pct} from the paper. 0 to disable.')

# How much to optimize
parser.add_argument('--lr-policy', type = str, default = 'constant', choices = LR_POLICY_CHOICES,
help = 'Learning rate policy. See description in lr-params.')
parser.add_argument('--lr-params', type = str, default = '{"lr": 1}',
help = 'Learning rate params, specified as a string that evalutes to a Python dict. Params that must be provided dependon which lr-policy is selected. The "constant" policy requires the "lr" key and uses the constant given learning rate. The "progress" policy requires the "max_lr" and "desired_prog" keys and scales the learning rate such that the objective function will change by an amount equal to DESIRED_PROG under a linear objective assumption, except the LR is limited to MAX_LR. The "progress01" policy requires the "max_lr", "early_prog", and "late_prog_mult" keys and is tuned for optimizing neurons with outputs in the [0,1] range, e.g. neurons on a softmax layer. Under this policy optimization slows down as the output approaches 1 (see code for details).')
parser.add_argument('--max-iter', type = int, default = 500,
help = 'Number of iterations of the optimization loop.')

# Where to save results
parser.add_argument('--output-prefix', type = str, default = 'optimize_results/opt',
help = 'Output path and filename prefix (default: optimize_results/opt)')
parser.add_argument('--output-template', type = str, default = '%(p.push_layer)s_%(p.push_channel)04d_%(p.rand_seed)d',
help = 'Output filename template; see code for details (default: "%%(p.push_layer)s_%%(p.push_channel)04d_%%(p.rand_seed)d"). '
'The default output-prefix and output-template produce filenames like "optimize_results/opt_prob_0278_0_best_X.jpg"')
parser.add_argument('--brave', action = 'store_true', help = 'Allow overwriting existing results files. Default: off, i.e. cowardly refuse to overwrite existing files.')
parser.add_argument('--skipbig', action = 'store_true', help = 'Skip outputting large *info_big.pkl files (contains pickled version of x0, last x, best x, first x that attained max on the specified layer.')

args=parser.parse_args()



print(os.path.exists('/p/work/rditljtd'))
# path = '/home/jordan/Development/Thesis/MaterialClassification/minc-model/'
path = '/home/jordan/Development/Thesis/deep-visualization-toolbox/models/minc-googlenet/'
if not os.path.exists(path + 'images'):
    print('Place images to be classified in images/brick/*.jpg, images/carpet/*.jpg, ...')
    exit()
    #     sys.exit(1)
# categories=[x.strip() for x in open(path + 'categories.txt').readlines()]
arch='googlenet' # googlenet, vgg16 or alexnet
print(path+'deploy-{}.prototxt'.format(arch), path + 'minc-{}.caffemodel'.format(arch))
net1=caffe.Classifier(path + 'deploy-{}.prototxt'.format(arch), path + 'minc-{}.caffemodel'.format(arch),channel_swap=(2,1,0),mean=numpy.array([104,117,124]))
# net2=caffe.Classifier(path + 'deploy-{}-conv.prototxt'.format(arch), path + 'minc-{}.caffemodel'.format(arch),channel_swap=(2,1,0),mean=numpy.array([104,117,124]))

# layer_name = 'fc8-20'
# layer_name = 'fc8-20'
# node_number = 10

for arg in vars(args):
    print(arg)
    print('here')
    print(getattr(args, arg), parser.get_default(arg))
    if getattr(args, arg) == parser.get_default(arg):
        setattr(args, arg,None)

# layers = net1.layers
layer_name = 'inception_4d/pool'


node_count = 0
spatial = "(7,7)"

for node in (net1.blobs[layer_name].data[0]):
    if node_count == len(net1.blobs[layer_name].data[0]):
        break
    command = construct_command(None, None, None, None, None, args.data_size, None, args.rand_seed, 
                layer_name, node_count, spatial, args.push_dir, args.decay,
                args.blur_radius, args.blur_every, args.small_val_percentile, args.small_norm_percentile,
                args.px_benefit_percentile, args.px_abs_benefit_percentile, args.lr_policy,
                args.lr_params, args.max_iter, args.output_prefix, args.output_template,
                args.brave, args.skipbig)
    os.system(command)
    node_count += 1



# for layer_name in net1._layer_names:
    # construct_command()