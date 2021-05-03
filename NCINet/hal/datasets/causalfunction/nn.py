# nn.py

import torch
import torch.nn as nn
import random
import numpy as np

__all__ = ['NN']

def scale_tensor(data):
    mean=data.mean()
    std=data.std()

    return (data-mean)/std

def normal_noise(nsamples,ndims,nranges):

    output=torch.rand(1)*torch.randn(nsamples,ndims)+ torch.tensor(random.sample([nranges, -nranges], 1))

    return output

class NN(nn.Module):

    def __init__(self, indim=8,outdim=8,hdlayers=10,nlayers=1):
        super().__init__()
        
        self.indim=indim
        self.outdim=outdim
        # self.hdlayers=np.random.randint(8, 13)
        # self.nlayers=np.random.randint(0, 3)

        layers=[]
        layers.append(nn.Linear(indim+1, hdlayers))
        layers.append(nn.ReLU())

        for l in range(nlayers):
            layers.append(nn.Linear(hdlayers, hdlayers))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hdlayers, outdim))

        self.model = nn.Sequential(*layers)

    def forward(self, cause):
        #cause = torch.tensor(cause,dtype=torch.float32)
        self.noise = normal_noise(cause.shape[0],1,2)
        cause=torch.cat((cause,self.noise),1)
        with torch.no_grad():
            effect=self.model(cause)
        #import pdb;pdb.set_trace()
        return scale_tensor(effect)