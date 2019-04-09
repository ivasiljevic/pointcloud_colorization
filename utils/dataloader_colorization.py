##############################################################
# faro_dataloader.py
# Creates a pytorch dataloader FARO3D
# FARO3D(data.Dataset)
##############################################################

import numpy as np

import os
import collections
import json
from PIL import Image

import torch
from torch.utils import data
import pdb
import glob
from torch.utils.data import DataLoader
from skimage.color import rgb2lab

MODEL_TYPE = '_nofloor'

class FARO3D(data.Dataset):
    def __init__(self, root='../dataset/lores/*1024.txt'):
        self.root = root
        #self.root = '../dataset/lores/*1024.txt'
        #self.img_transform = img_transform
        #self.label_transform = label_transform
        temp = glob.glob(root)
        self.im_list = [tmp.split('/')[-1] for tmp in temp]

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        high_res = np.loadtxt("../dataset" + MODEL_TYPE + "/lores/" + self.im_list[index])
        high_res = np.swapaxes(high_res, 1, 0)

        yaw_rad = np.random.uniform(0, np.pi)
        # for now just try to fit dataset.
        R = rot_yaw(yaw_rad)
        tmp = high_res[:3,:].T
        tmp = np.dot(tmp, R)
        high_res[:3,:] = tmp.T

        high_res[3:,:] = high_res[3:,:]/255.
        low_res = high_res[0:3,:]
        high_res = high_res[3:,:]
        tmp = high_res.T
        tmp = np.reshape(tmp, (4, int(high_res.shape[1]/4), 3))
        tmp = rgb2lab(tmp)
        tmp = np.reshape(tmp, (high_res.shape[1], 3))
        high_res = tmp.T
        sample = [torch.from_numpy(low_res), torch.from_numpy(high_res)]

        return sample

def rot_yaw(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
