# hadamard.py
import torch
import torch.nn as nn


__all__ = ['HadamardFun']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

def normal_noise(nsamples,ndims):

    output=0.1*torch.rand(1)*torch.randn(nsamples,ndims)

    return output

class HadamardFun(nn.Module):
    def __init__(self,indim,outdim,indim2=0):
        super(HadamardFun, self).__init__()
        self.indim=indim
        self.outdim=outdim
        self.indim2=indim2


    def forward(self, cause,cause2=None):
        assert self.indim==self.outdim,'ndimx and ndimy should be same in HadamardFun'


        A=torch.randn(cause.shape[0],cause.shape[1])
        B = torch.randn(cause.shape[0], cause.shape[1])
        effect=A*cause*cause+B*cause+normal_noise(cause.shape[0],cause.shape[1])

        if cause2 is not None:

            effect2=A*cause2*cause2+B*cause2
            effect=effect+effect2

        return scale_tensor(effect)