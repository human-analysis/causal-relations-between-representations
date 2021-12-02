# classification.py

__all__ = ['Top1Classification']

class Top1Classification:
    def __init__(self):
        pass

    def __call__(self, output, target):
        batch_size = target.size(0)

        pred = output.data.max(1)[1].view(-1, 1).squeeze()
        res = pred.eq(target.data).cpu().sum().float() * 100 / batch_size

        return res
