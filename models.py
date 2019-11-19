import torch
from torch import nn
from torch.functional import F
from fastai.text import *

class Model2(nn.Module):
    def __init__(self, nh, nv=31):
        super().__init__()
        self.nh = nh
        self.nv = nv
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        
    def forward(self, x):
        h = torch.zeros(x.shape[0],  self.nh).to(device=x.device)
        res = []
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.h_o(self.bn(h)))
        return torch.stack(res, dim=1)

class Model3(nn.Module):
    def __init__(self, nh, bs, nv=31):
        super().__init__()
        self.nh = nh
        self.nv = nv
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).cuda()
        
    def forward(self, x):
        res = []
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = h.detach()
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res

class mGRU(nn.Module):
    def __init__(self, nh, bs, layers=2, nv=31):
        super().__init__()
        self.nh = nh
        self.nv = nv
        self.layers = layers
        self.bs = bs
        self.i_h = nn.Embedding(nv,nh)
        self.cell = nn.GRU(nh, nh, layers, batch_first=True)
        self.h_o = nn.Linear(nh,nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(layers, bs, nh).cuda()
        
    def forward(self, x):
        res,h = self.cell(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))

class mRNN(nn.Module):
    def __init__(self, nh, bs, layers=2, nv=31):
        super().__init__()
        self.nh = nh
        self.nv = nv
        self.layers = layers
        self.bs = bs
        self.i_h = nn.Embedding(nv,nh)
        self.cell = nn.RNN(nh,nh, layers, batch_first=True)
        self.h_o = nn.Linear(nh,nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(layers, bs, nh).cuda()
        
    def forward(self, x):
        res,h = self.cell(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))