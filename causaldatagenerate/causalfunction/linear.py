# linear.py
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

__all__ = ['LinearFun']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

def normal_noise(nsamples,ndims,nranges):

    output=0.1*torch.rand(1)*torch.randn(nsamples,ndims)
    return output
class LinearFun(nn.Module):
    def __init__(self,indim,outdim,indim2=0):
        super(LinearFun, self).__init__()
        self.indim=indim
        self.outdim=outdim
        self.indim2=indim2


    def forward(self, cause,cause2=None):
  
        if cause2 is not None:
            cause=torch.cat((cause,cause2),1)
            A=torch.randn(self.indim+self.indim2,self.outdim)
            effect=cause.mm(A)+normal_noise(cause.shape[0],self.outdim,1)
        else:
            A=torch.randn(self.indim,self.outdim)
            effect=cause.mm(A)+normal_noise(cause.shape[0],self.outdim,1)

        return scale_tensor(effect)