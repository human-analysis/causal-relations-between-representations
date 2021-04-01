
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import interpolate

__all__ = ['CausalGenerator']

def scale_tensor(data):
    mean=data.mean()
    std=data.std()

    return (data-mean)/std


def normal_init(nsamples,ndims,nranges):

    #output=torch.rand(1,dtype=torch.float64)*torch.randn(nsamples,ndims,dtype=torch.float64)+ torch.tensor(random.sample([nranges, -nranges], 1))
    output=np.random.rand(1) * np.random.randn(nsamples, ndims)+ random.sample([2, -2], 1)
    return scale_tensor(output)

def normal_noise(nsamples,ndims,nranges):

    output=torch.rand(1)*torch.randn(nsamples,ndims)+ torch.tensor(random.sample([nranges, -nranges], 1))

    return output

class LinearFun(nn.Module):
    def __init__(self):
        super(LinearFun, self).__init__()


    def forward(self, cause):
        cause = torch.tensor(cause)
        A=torch.rand(cause.shape[0],cause.shape[0])
        effect=A.mm(cause)+normal_noise(cause.shape[0],cause.shape[1],1)

        return scale_tensor(effect)

class QuadraticFun(nn.Module):
    def __init__(self):
        super(QuadraticFun, self).__init__()


    def forward(self, cause):
        cause = torch.tensor(cause)
        A=torch.rand(cause.shape[0],cause.shape[1])
        B=torch.rand(cause.shape[0],cause.shape[0])
        effect=A.mm(cause.T.mm(cause))+B.mm(cause)+normal_noise(cause.shape[0],cause.shape[1],1)

        return scale_tensor(effect)

class HadamardFun(nn.Module):
    def __init__(self):
        super(HadamardFun, self).__init__()


    def forward(self, cause):

        A=torch.rand(cause.shape[0],cause.shape[1])
        B = torch.rand(cause.shape[0], cause.shape[1])
        effect=A*cause*cause+B*cause+normal_noise(cause.shape[0],cause.shape[1],1)

        return scale_tensor(effect)

class BilinearFun(nn.Module):
    def __init__(self):
        super(BilinearFun, self).__init__()


    def forward(self, cause):
        cause=torch.tensor(cause)
        m = nn.Bilinear(8, 8, 8,bias=False)
        effect=m(cause,cause)+normal_noise(cause.shape[0],cause.shape[1],1)

        return scale_tensor(effect)

class GPFun(nn.Module):
    def __init__(self):
        super(GPFun, self).__init__()

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

class Cubicspline(nn.Module):
    def __init__(self):
        super(Cubicspline, self).__init__()

    def forward(self, cause):
        effect=np.zeros(cause.shape)
        for dim in range(cause.shape[1]):
            x=cause[:,dim]
            dKnot = np.random.randint(4, 5)
            supportX = sorted(np.random.randn(dKnot))
            effect[:,dim]=interpolate.PchipInterpolator(supportX, np.random.randn(dKnot))(x)
        return scale_tensor(effect)

class NN(nn.Module):

    def __init__(self, indim=8,outdim=8,hdlayers=10,nlayers=1):
        super().__init__()

        layers=[]
        layers.append(nn.Linear(indim+1, hdlayers))
        layers.append(nn.ReLU())

        for l in range(nlayers):
            layers.append(nn.Linear(hdlayers, hdlayers))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hdlayers, outdim))

        self.model = nn.Sequential(*layers)

    def forward(self, cause):
        cause = torch.tensor(cause,dtype=torch.float32)
        self.noise = normal_noise(cause.shape[0],1,2)
        cause=torch.cat((cause,self.noise),1)
        effect=self.model(cause)

        return scale_tensor(effect)


class CausalGenerator():
    def __init__(self, causal_fun):
        super(CausalGenerator, self).__init__()
        self.mechanism = {'linear': LinearFun,
                          'quadratic': QuadraticFun,
                          'hadamard':HadamardFun,
                          'bilinear':BilinearFun,
                          'GP':GPFun,
                          'cubicspline':Cubicspline,
                          'nn':NN,
                        }[causal_fun]
        self.initial_data=normal_init

    def generator(self,nsamples,ndims,nranges):
        # initial cause

        cause=self.initial_data(nsamples,ndims,nranges)
        causal_mechanism=self.mechanism()
        effect=causal_mechanism(cause)

        return effect