# hadamard.py
import torch
import torch.nn as nn
import random

__all__ = ['HadamardFun']

def scale_tensor(data):
    mean=data.mean()
    std=data.std()

    return (data-mean)/std

def normal_noise(nsamples,ndims,nranges):

    output=torch.rand(1)*torch.randn(nsamples,ndims)+ torch.tensor(random.sample([nranges, -nranges], 1))

    return output

class HadamardFun(nn.Module):
    def __init__(self,indim,outdim):
        super(HadamardFun, self).__init__()
        self.indim=indim
        self.outdim=outdim


    def forward(self, cause):
        assert self.indim==self.outdim,'ndimx and ndimy should be same in HadamardFun'
        cause = torch.tensor(cause,dtype=torch.float32)
        A=torch.rand(cause.shape[0],cause.shape[1])
        B = torch.rand(cause.shape[0], cause.shape[1])
        effect=A*cause*cause+B*cause+normal_noise(cause.shape[0],cause.shape[1],1)
        #import pdb;pdb.set_trace()
        return scale_tensor(effect)