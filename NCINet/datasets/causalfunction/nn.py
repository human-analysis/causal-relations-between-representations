# nn.py
import torch
import torch.nn as nn
import numpy as np

__all__ = ['NN']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

def normal_noise(nsamples,ndims):

    output=0.1*torch.rand(1)*torch.randn(nsamples,ndims)

    return output

class NN(nn.Module):

    def __init__(self, indim1=8,outdim=8,indim2=0,hdlayers=10,nlayers=1):
        super().__init__()
        
        self.indim1=indim1
        self.outdim=outdim
        self.hdlayers=np.random.randint(8, 20)
        self.nlayers=np.random.randint(0, 3)

        layers=[]
        layers.append(nn.Linear(indim1+indim2+1, hdlayers))
        layers.append(nn.ReLU())

        for l in range(nlayers):
            layers.append(nn.Linear(hdlayers, hdlayers))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hdlayers, outdim))

        self.model = nn.Sequential(*layers)

    def forward(self, cause1,cause2=None):

        if cause2 is not None:
            self.noise = normal_noise(cause1.shape[0],1)
            cause1=torch.cat((cause1,cause2),1)
            cause1=torch.cat((cause1,self.noise),1)
            with torch.no_grad():
                effect=self.model(cause1)
        else:
            self.noise = normal_noise(cause1.shape[0],1)
            cause1=torch.cat((cause1,self.noise),1)
            with torch.no_grad():
                effect=self.model(cause1)

        return scale_tensor(effect)