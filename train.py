import os.path as osp
import pdb
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from data.Indoor3DSemSegLoader import Indoor3DSemSeg
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import PointConv, fps, radius, GraphConv, SGConv, GMMConv
from torch_geometric.transforms import Polar
import numpy as np
from pdb import set_trace as bp
from torch_geometric.nn import EdgeConv, knn_graph, global_max_pool

REPORT_RATE = 100
NUM_POINTS = 1024

#pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

dset = Indoor3DSemSeg(NUM_POINTS, "./")
train_loader = DataLoader(dset, batch_size=32, shuffle=True)
val_loader =   DataLoader(dset, batch_size=1, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        nn = Seq(Lin(6, 64), ReLU(), Lin(64, 64), ReLU(), Lin(64, 64), ReLU())
        self.conv1 = EdgeConv(nn, aggr='max')

        nn = Seq(
            Lin(128, 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, 256),
            ReLU())
        self.conv2 = EdgeConv(nn, aggr='max')

        self.lin0 = Lin(256, 512)

        self.lin1 = Lin(832, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, data):


        t, pos, batch = data.x, data.pos, data.batch
        pos = pos.cuda()
        batch = batch.cuda()
        t = t.cuda()

        #edge_index = data.edge_index
        # pos = pos.double()
        # batch = batch.long()
        dsize = pos.size()[0]
        bsize = batch[-1].item() + 1
        edge_index = knn_graph(pos, k=30, batch=batch)
        x1 = self.conv1(pos, edge_index)

        edge_index = knn_graph(x1, k=30, batch=batch)
        x2 = self.conv2(x1, edge_index)

        x2max = F.relu(self.lin0(x2))

        x2max = global_max_pool(x2max, batch)
        globalfeats = x2max.repeat(1,int(dsize/bsize)).view(dsize, x2max.size()[1])

        concat_features = torch.cat((x1,x2, globalfeats), dim=1)

        x = F.relu(self.lin1(concat_features))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = Net(23)#.float()#to(device)
num_classes = 23
model = Net(23).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    count = 0
    model.train()


    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        seg_pred = model(data)

        loss = F.nll_loss(seg_pred, data.y)

        if count % REPORT_RATE == 0:
            print(epoch, loss.data.cpu(),flush=True)
        loss.backward()
        optimizer.step()
        count = count + 1



def test(loader):
    model.eval()
    correct_type = 0
    correct_split = 0

    pt_size = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            #pred = model(data).max(1)[1]
            pred = model(data)
            pred = pred.max(1)[1].long()

        #print(data.y.shape)
        correct_type += pred.eq(data.y.long()).sum().item()
        pt_size += pred.shape[0]
        #print(split_pred.eq(data.y[1,:].long()).sum().item())
    return correct_type / pt_size


for epoch in range(1, 201):
    train(epoch)
    val_acc = test(val_loader)
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, val_acc))
