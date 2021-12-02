# classification.py

from torch import nn

__all__ = ['Classification']


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss
