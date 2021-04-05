# cubicspline.py

import torch
import torch.nn as nn
import random
import numpy as np
from scipy import interpolate

__all__ = ['Cubicspline']

def scale_tensor(data):
    mean=data.mean()
    std=data.std()

    return (data-mean)/std

class Cubicspline(nn.Module):
    def __init__(self,indim,outdim):
        super(Cubicspline, self).__init__()
        self.indim=indim
        self.outdim=outdim

    def forward(self, cause):
        assert self.indim==self.outdim,'ndimx and ndimy should be same in Cubicspline'
        effect=np.zeros(cause.shape)
        for dim in range(cause.shape[1]):
            x=cause[:,dim]
            dKnot = np.random.randint(4, 5)
            supportX = sorted(np.random.randn(dKnot))
            effect[:,dim]=interpolate.PchipInterpolator(supportX, np.random.randn(dKnot))(x)
        return scale_tensor(torch.tensor(effect))