# bilinear.py
import torch
import torch.nn as nn
import random

__all__ = ['BilinearFun']

def scale_tensor(data):
    mean=data.mean()
    std=data.std()

    return (data-mean)/std

def normal_noise(nsamples,ndims,nranges):

    output=torch.rand(1)*torch.randn(nsamples,ndims)+ torch.tensor(random.sample([nranges, -nranges], 1))

    return output

class BilinearFun(nn.Module):
    def __init__(self,indim,outdim):
        super(BilinearFun, self).__init__()
        self.indim=indim
        self.outdim=outdim


    def forward(self, cause):
        #cause = torch.tensor(cause,dtype=torch.float32)
        m = nn.Bilinear(self.outdim, self.indim, self.indim,bias=False)
        #import pdb;pdb.set_trace()
        with torch.no_grad():
            effect=m(cause,cause)+normal_noise(cause.shape[0],self.outdim,1)
        #import pdb;pdb.set_trace()

        return scale_tensor(effect)