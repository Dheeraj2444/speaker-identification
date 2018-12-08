SERVER = False

# Files and Directories
ROOT_DIR = ""
TRAIN_PATH = 'wav_train_subset'
STFT_FOLDER = 'stft'
CHECKPOINTS_FOLDER = "checkpoints"
PAIRS_FILE = 'pairs.csv'
VGG_VOX_WEIGHT_FILE = "vggvox_ident_net.mat"

# Data_Part
TOTAL_USERS = 100
CLIPS_PER_USER = 10
MIN_CLIP_DURATION = 3.

# ML_Part
TRAINING_USERS = 80
SIMILAR_PAIRS = 20
DISSIMILAR_PAIRS = SIMILAR_PAIRS

LEARNING_RATE = 5e-4
N_EPOCHS = 30
BATCH_SIZE = 32

assert SIMILAR_PAIRS <= CLIPS_PER_USER * (CLIPS_PER_USER - 1)


from tqdm import tqdm
import os
import sys
import time
import itertools
from collections import Counter
from IPython.core.display import HTML
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy
import sklearn
import librosa
import librosa.display
import wave
import contextlib
import matplotlib.pyplot as plt
from collections import OrderedDict
# import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint

assert os.path.exists(STFT_FOLDER)
assert os.path.exists(CHECKPOINTS_FOLDER)


def get_rel_path(path, server=SERVER, root_dir=ROOT_DIR):
    if server:
        return os.path.join(root_dir, path)
    else:
        return path


def wavPlayer(filepath):
    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>

    <body>
    <audio controls="controls" style="width:600px" >
      <source src="%s" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """%(filepath)
    display(HTML(src))
    
    
def load_pretrained_weights():
    weights = {}

    # loading pretrained vog_vgg learned weights
    vox_weights = loadmat(get_rel_path(VGG_VOX_WEIGHT_FILE), 
                          struct_as_record=False, squeeze_me=True)

    for l in vox_weights['net'].layers[:-1]:
        if len(l.weights) > 0:
            weights[l.name] = l.weights
    #         print(l.name, [i.shape for i in l.weights])

    for i in weights:
        weights[i][0] = weights[i][0].T 

    weights['conv1'][0] = np.expand_dims(weights['conv1'][0], axis=1)
    weights['fc6'][0] = np.expand_dims(weights['fc6'][0], axis=3)
    weights['fc7'][0] = np.expand_dims(weights['fc7'][0], axis=-1)
    weights['fc7'][0] = np.expand_dims(weights['fc7'][0], axis=-1)

#     print(weights.keys())   
#     for key in weights:
#         print(key, [i.shape for i in weights[key]])
    return weights


# parameters
conv_kernel1, n_f1, s1, p1 = 7, 96, 2, 1
pool_kernel1, pool_s1 = 3, 2

conv_kernel2, n_f2, s2, p2 = 5, 256, 2, 1
pool_kernel2, pool_s2 = 3, 2

conv_kernel3, n_f3, s3, p3 = 3, 384, 1, 1

conv_kernel4, n_f4, s4, p4 = 3, 256, 1, 1

conv_kernel5, n_f5, s5, p5 = 3, 256, 1, 1
pool_kernel5_x, pool_kernel5_y, pool_s5_x, pool_s5_y = 5, 3, 3, 2

conv_kernel6_x, conv_kernel6_y, n_f6, s6 = 9, 1, 4096, 1

conv_kernel7, n_f7, s7 = 1, 1024, 1

conv_kernel8, n_f8, s8 = 1, 1024, 1


def save_checkpoint(state, loss):
    """Save checkpoint if a new best is achieved"""
    fname = "checkpoint_" + time.strftime("%Y%m%d-%H%M%S") + "_" + str(loss.item()) + ".pth.tar"
    torch.save(state, get_rel_path(os.path.join(CHECKPOINTS_FOLDER, fname)))  # save checkpoint
    print("$$$ Saved a new checkpoint\n")