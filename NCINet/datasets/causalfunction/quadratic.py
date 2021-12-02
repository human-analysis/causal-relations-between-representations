# quadratic.py
import torch
import torch.nn as nn

__all__ = ['QuadraticFun']

def scale_tensor(data):
    mean=data.mean(0)
    std=data.std(0)

    return (data-mean)/(std+1e-5)

def normal_noise(nsamples,ndims):

    output=0.1*torch.rand(1)*torch.randn(nsamples,ndims)

    return output

class QuadraticFun(nn.Module):
    def __init__(self,indim1,outdim,indim2=0):
        super(QuadraticFun, self).__init__()
        self.indim1=indim1
        self.outdim=outdim
        self.indim2=indim2

    def forward(self, cause1,cause2=None):

        A=torch.randn(cause1.shape[0],self.outdim)
        B=torch.randn(self.indim1,self.outdim)
        effect=(cause1.mm(cause1.T)).mm(A)+cause1.mm(B)+normal_noise(cause1.shape[0],self.outdim)
        if cause2 is not None:
            effect=effect+(cause2.mm(cause2.T)).mm(A)+cause2.mm(B)

        return scale_tensor(effect)





