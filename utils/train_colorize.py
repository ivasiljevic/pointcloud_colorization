import torch
import torch.nn as nn
import torch.autograd as grad
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import time
import os.path as osp
import os, sys
sys.path.append("/share/data/vision-greg/proj/ivas/zero_shot")
from pointnet_1x_new import PointNetUpsampler
from dataloader_colorization import FARO3D

#MODEL_TYPE = "4x"
MODEL_TYPE = "nofloor"

def main():
    

    batch_size = 24
    num_epochs = 200
    lr = 0.001
    printout = 20

    #try:
    #    os.mkdir(snapshot_dir)
    #except:
    #    pass
    # Instantiate a dataset loader
    #model_net = ModelNet40(dataset_root_path)
    #data_loader = DataLoader(model_net, batch_size=batch_size,
    #    shuffle=True, num_workers=12)
    #gt_key = model_net.get_gt_key()

    # Instantiate the network
    #classifier = PointNetClassifier(num_points, dims).train().cuda().double()
    faro_dataset_1024 = FARO3D(root='../dataset_' + MODEL_TYPE + '/lores/*1024.txt')
    faro_dataset_2048 = FARO3D(root='../dataset_' + MODEL_TYPE + '/lores/*2048.txt')
    faro_dataset_4096 = FARO3D(root='../dataset_' + MODEL_TYPE + '/lores/*4096.txt')
    faro_dataset_8192 = FARO3D(root='../dataset_' + MODEL_TYPE + '/lores/*8192.txt')

    class ConcatDataset(torch.utils.data.Dataset):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.datasets)

        def __len__(self):
            return min(len(d) for d in self.datasets)

    #pc_type = np.random.choice([1024, 2048, 4096, 8192])

    data_loader = torch.utils.data.DataLoader(
        ConcatDataset(
            faro_dataset_1024,
            faro_dataset_2048,
            faro_dataset_4096,
            faro_dataset_8192),
        batch_size = batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    classifier = PointNetUpsampler().train().cuda().double()

    loss = nn.MSELoss()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)



    # Some timers and a counter
    forward_time = 0.
    backprop_time = 0.
    network_time = 0.
    batch_counter = 0

    # Whether to save a snapshot
    save = False
    criterion = nn.MSELoss()

    # Run through all epochs
    for ep in range(num_epochs):
        ep = ep + 1

        if ep % 5 == 0:
            print("Saving model...")
            save_model(classifier, ep)
        # Update the optimizer according to the learning rate schedule
        scheduler.step()


        idx = np.random.choice([0, 1, 2, 3])
        #data_loader_ = data_loader[idx]

        for i, samples in enumerate(data_loader):
            for sample in samples:
                # Parse loaded data
                points = grad.Variable(sample[0]).cuda()
                target = grad.Variable(sample[1]).cuda()

                # Record starting time
                start_time = time.time()

                # Zero out the gradients
                optimizer.zero_grad()

                # Forward pass
                pred = classifier(points)
                #print(pred.shape, points.shape)

                # Compute forward pass time
                forward_finish = time.time()
                forward_time += forward_finish - start_time

                pred_error = criterion(pred, target)

                # Backpropagate
                pred_error.backward()

                # Update the weights
                optimizer.step()

                if batch_counter % 20 == 0:
                    print("Current (epoch, loss): ({}, {})".format(ep, pred_error), flush=True)
                    print("Pred: {}, True: {}".format(pred[0,:,0].data, target[0,:,0].data, flush=True))

                batch_counter += 1

def save_model(model, ep):
    torch.save(model.state_dict(), "./models/"  + str(ep) + "_" + MODEL_TYPE + "_colorize_original_pointnet.params")   



if __name__ == '__main__':
    main()
