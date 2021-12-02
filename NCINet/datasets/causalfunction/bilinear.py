# bilinear.py
import torch
import torch.nn as nn


__all__ = ['BilinearFun']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

def normal_noise(nsamples,ndims):

    output=0.1*torch.rand(1)*torch.randn(nsamples,ndims)

    return output

class BilinearFun(nn.Module):
    def __init__(self,indim1,outdim,indim2=0):
        super(BilinearFun, self).__init__()
        self.indim1=indim1
        self.outdim=outdim
        self.indim2=indim2

    def forward(self, cause1,cause2=None):

        if cause2 is not None:
            m = nn.Bilinear(self.outdim, self.indim1, self.indim1,bias=False)
            with torch.no_grad():
                effect=m(cause1,cause1)+m(cause2,cause2)+normal_noise(cause1.shape[0],self.outdim)
        else:
            m = nn.Bilinear(self.outdim, self.indim1, self.indim1,bias=False)
            with torch.no_grad():
                effect=m(cause1,cause1)+normal_noise(cause1.shape[0],self.outdim)

        return scale_tensor(effect)