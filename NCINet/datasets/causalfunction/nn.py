# nn.py

import torch
import torch.nn as nn
import random
import numpy as np

__all__ = ['NN']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

def normal_noise(nsamples,ndims,nranges):

    output=0.1*torch.rand(1)*torch.randn(nsamples,ndims)

    return output

class NN(nn.Module):

    def __init__(self, indim=8,outdim=8,indim2=0,hdlayers=10,nlayers=1):
        super().__init__()
        
        self.indim=indim
        self.outdim=outdim
        self.hdlayers=np.random.randint(8, 20)
        self.nlayers=np.random.randint(0, 3)

        layers=[]
        layers.append(nn.Linear(indim+1+indim2, hdlayers))
        layers.append(nn.ReLU())

        for l in range(nlayers):
            layers.append(nn.Linear(hdlayers, hdlayers))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hdlayers, outdim))

        self.model = nn.Sequential(*layers)

    def forward(self, cause,cause2=None):

        if cause2 is not None:
            self.noise = normal_noise(cause.shape[0],1,2)
            cause=torch.cat((cause,cause2),1)
            cause=torch.cat((cause,self.noise),1)
            with torch.no_grad():
                effect=self.model(cause)
        else:

            self.noise = normal_noise(cause.shape[0],1,2)
            cause=torch.cat((cause,self.noise),1)
            with torch.no_grad():
                effect=self.model(cause)

        return scale_tensor(effect)