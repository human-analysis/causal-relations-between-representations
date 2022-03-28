# cubicspline.py

import torch
import torch.nn as nn
import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

__all__ = ['Cubicspline']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

class Cubicspline(nn.Module):
    def __init__(self,indim,outdim,indim2=0):
        super(Cubicspline, self).__init__()
        self.indim=indim
        self.outdim=outdim
        self.indim2=indim2

    def forward(self, cause,cause2=None):
        assert self.indim==self.outdim,'ndimx and ndimy should be same in Cubicspline'

        cause=cause.numpy()
        effect=np.zeros(cause.shape)
        dKnot = np.random.randint(5,20)
        supportX =sorted(np.random.randn(dKnot))
        Y=np.random.randn(dKnot)
        for dim in range(cause.shape[1]):
            x=cause[:,dim]        
            effect[:,dim]=interpolate.PchipInterpolator(supportX, Y)(x)
         
        if cause2 is not None:
            cause2=cause2.numpy()
            effect2=np.zeros(cause2.shape)
            for dim in range(cause.shape[1]):
                x=cause2[:,dim]
                effect2[:,dim]=interpolate.PchipInterpolator(supportX, Y)(x)
            effect=effect+effect2

        return scale_tensor(torch.tensor(effect,dtype=torch.float))