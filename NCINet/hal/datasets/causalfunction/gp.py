# gp.py
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from torch.distributions.multivariate_normal import MultivariateNormal

__all__ = ['GPFun']

def scale_tensor(data):
    mean=data.mean()
    std=data.std()

    return (data-mean)/std

def normal_noise(nsamples,ndims,nranges):

    output=torch.rand(1)*torch.randn(nsamples,ndims)+ torch.tensor(random.sample([nranges, -nranges], 1))

    return output

class GPFun(nn.Module):
    def __init__(self,indim,outdim):
        super(GPFun, self).__init__()
        self.indim=indim
        self.outdim=outdim

    def forward(self, cause):
        org_shape=cause.shape
        cause=np.reshape(cause, (-1, 1))
        xnorm = np.power(euclidean_distances(cause, cause), 2)#[:,np.newaxis]
        xnorm= np.exp(-xnorm / (2.0))
        mean = np.zeros(cause.shape[0])
        effect = np.random.multivariate_normal(mean, xnorm)
        effect = np.reshape(effect, org_shape)

        # output = torch.rand(1,dtype=torch.float64) * torch.randn(1000, 1,dtype=torch.float64) + torch.tensor(random.sample([2, -2], 1))
        # cause=output.numpy()
        # cov = torch.pow(torch.cdist(cause.unsqueeze(1), cause.unsqueeze(1),p=2), 2)
        # cov=torch.exp(-cov / (2.0))
        # mean = torch.zeros(cause.shape,dtype=torch.float64)
        # effect = MultivariateNormal(torch.tensor(mean), torch.tensor(xnorm))

        return scale_tensor(torch.tensor(effect))