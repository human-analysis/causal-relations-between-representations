# proj_grad.py

import torch

__all__ = ['Projection_gauss']


class Projection_gauss:
    def __init__(self):
        pass

    def __call__(self, Z, S, Y, reg, device, sigma):

        batch_size = Z.size(0)
        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z,dim=1).reshape([1,batch_size])
        NORM = NORM ** 2

        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM)+2*torch.mm(Z, torch.t(Z)))/sigma) # Z is already transposed

        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        P1 = torch.mm(torch.t(M), M)
        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        S_bar = S - torch.mean(S, dim=0)

        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2


        return Project_S

