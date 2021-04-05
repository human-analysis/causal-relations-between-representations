# quadratic.py
import torch
import torch.nn as nn
import random

__all__ = ['QuadraticFun']

def scale_tensor(data):
    mean=data.mean()
    std=data.std()

    return (data-mean)/std

def normal_noise(nsamples,ndims,nranges):

    output=torch.rand(1)*torch.randn(nsamples,ndims)+ torch.tensor(random.sample([nranges, -nranges], 1))

    return output

class QuadraticFun(nn.Module):
    def __init__(self,indim,outdim):
        super(QuadraticFun, self).__init__()
        self.indim=indim
        self.outdim=outdim


    def forward(self, cause):
        cause = torch.tensor(cause,dtype=torch.float32)
        A=torch.rand(cause.shape[0],self.outdim)
        B=torch.rand(self.indim,self.outdim)
        #import pdb;pdb.set_trace()
        effect=(cause.mm(cause.T)).mm(A)+cause.mm(B)+normal_noise(cause.shape[0],self.outdim,1)

        return scale_tensor(effect)





