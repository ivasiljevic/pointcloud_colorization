from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex
from torch_geometric.data import Data, DataLoader
import pptk


CLOUD_LIST = np.genfromtxt('color_clouds.txt',dtype='str')


class DIODE(data.Dataset):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

    def __getitem__(self, idx):

        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(np.loadtxt(CLOUD_LIST[idx])[pt_idxs].copy()).type(
            torch.FloatTensor
        )

        current_labels = current_points[:,3:]

        data = Data(y=current_labels, pos=current_points[:,:3], x=current_points[:,:3])

        #return current_points, current_labels
        return data


    def __len__(self):
        return len(CLOUD_LIST)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = DIODE(1024)
    #dset = Indoor3DSemSeg(16, "./", train=True)
    print(dset[0])
    print(len(dset))
    dloader = DataLoader(dset, batch_size=1, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data.pos, data.y
        print(inputs.shape, labels.shape)
        #v = pptk.viewer(inputs.numpy())
        #v.set(point_size=0.1)
        #print("Saving..")
        #v.capture('./figures/screenshot.png')
        #wreck()
        if i == len(dloader) - 1:
            print(inputs.shape)
