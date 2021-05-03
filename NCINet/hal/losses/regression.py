# regression.py

from torch import nn

__all__ = ['Regression','highdim_Regression']


class Regression(nn.Module):

    def __init__(self):
        super(Regression, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, inputs, target):
        loss = self.loss(inputs, target)

        return loss
class highdim_Regression(nn.Module):

    def __init__(self):
        super(highdim_Regression, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, loss_x,loss_y):
  
        return loss_x.mean()+loss_y.mean()

