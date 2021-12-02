# cubicspline.py
import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate

__all__ = ['Cubicspline']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

class Cubicspline(nn.Module):
    def __init__(self,indim1,outdim,indim2=0):
        super(Cubicspline, self).__init__()
        self.indim1=indim1
        self.outdim=outdim
        self.indim2=indim2

    def forward(self, cause1,cause2=None):
        assert self.indim1==self.outdim,'ndimx and ndimy should be same in Cubicspline'

        cause1=cause1.numpy()
        effect=np.zeros(cause1.shape)
        dKnot = np.random.randint(5,20)
        supportX =sorted(np.random.randn(dKnot))
        Y=np.random.randn(dKnot)
        for dim in range(cause1.shape[1]):
            x=cause1[:,dim]
            effect[:,dim]=interpolate.PchipInterpolator(supportX, Y)(x)

        if cause2 is not None:
            cause2=cause2.numpy()
            effect2=np.zeros(cause2.shape)
            for dim in range(cause1.shape[1]):
                x=cause2[:,dim]
                effect2[:,dim]=interpolate.PchipInterpolator(supportX, Y)(x)
            effect=effect+effect2

        return scale_tensor(torch.tensor(effect,dtype=torch.float))