import os.path as osp
import pdb
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from data.Indoor3DSemSegLoader.py import Indoor3DSemSeg
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import PointConv, fps, radius, GraphConv, SGConv, GMMConv
from torch_geometric.transforms import Polar
import numpy as np

REPORT_RATE = 10

#pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)

dset = Indoor3DSemSeg(16, "./")
train_loader = DataLoader(dset, batch_size=32, shuffle=True)
#val_dataset = Skeletron(root = '/share/data/vision-greg/ivas/proj/skeletron/data')
#val_loader =   DataLoader(val_dataset, batch_size=1, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.gmm_conv1 = GMMConv(3, 32, dim=3) #pseudo-coordinate dim
        self.pt_conv2 = Lin(32, 64)

        self.lin1 = Lin(64, 32)
        self.lin2 = Lin(32, 16)
        self.lin3 = Lin(16, 4) # types
        self.lin4 = Lin(16, 3) # split or merge

    def forward(self, data):

        #batch = data.batch
        #data = Data(edge_index=data.edge_index, pos=data.pos)
        #data = Polar(norm=True)(data)
        x, pos, batch = data.x, data.pos, data.batch
        edge_index = data.edge_index#.cuda(0)
        pos = pos.double()#.do()
        batch = batch.long()

        x = F.relu(self.gmm_conv1(x.double(), edge_index, pos.double()))
        x = F.relu(self.pt_conv2(x))

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        type_pred = self.lin3(x)
        split_pred = self.lin4(x)
        return F.log_softmax(type_pred, dim=-1), F.log_softmax(split_pred, dim=-1)


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().double()#.float()#to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    count = 0
    model.train()


    for data in train_loader:
        #data = data.to(device)
        optimizer.zero_grad()

        type_pred, split_pred = model(data)

        split_gt = data.y[0, :].long()
        type_gt = data.y[1, :].long()
        if torch.sum(split_gt) == 0:
            continue
        else:
            print("type GT: ", torch.unique(split_gt))

        loss = F.nll_loss(split_pred, split_gt) + F.nll_loss(type_pred, type_gt)
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
        #data = data.to(device)
        with torch.no_grad():
            #pred = model(data).max(1)[1]
            type_pred, split_pred = model(data)

            type_pred = type_pred.max(1)[1].long()
            split_pred = split_pred.max(1)[1].long()

        #print(data.y.shape)
        correct_type += type_pred.eq(data.y[0,:].long()).sum().item()
        pt_size += split_pred.shape[0]
        #print(split_pred.eq(data.y[1,:].long()).sum().item())
        correct_split += split_pred.eq(data.y[1,:].long()).sum().item()
        #pdb.set_trace()
    return correct_type / pt_size, correct_split / pt_size

for epoch in range(1, 201):
    train(epoch)
    val_acc_type, val_acc_split = test(val_loader)
    print(val_acc_type, val_acc_split)
    #print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, val_acc))
