import torch.nn as nn
import config
import torch
import numpy as np

args = config.parse_args()

__all__ = ['Target','NN']

def scale_tensor(data):
        mean=data.mean(0)
        std=data.std(0)
    
        return (data-mean)/(std+1e-5)

def linear_regressor(x,y,reg):

    G=torch.zeros(x.shape[0],x.shape[2]+1,x.shape[2]+1).to("cuda")
    G[:,1:,1:]=np.sqrt(reg)*torch.eye(x.shape[2]).repeat(x.shape[0],1,1)
    X=torch.ones(x.shape[0],x.shape[1],x.shape[2]+1).to("cuda")
    X[:,:,1:]=x

    P1 = torch.bmm(torch.transpose(X,1,2), X)
    P2 = torch.inverse(P1+torch.bmm(torch.transpose(G,1,2), G) )
    P3 = torch.bmm(P2, torch.transpose(X,1,2))
    P4 = torch.bmm(torch.bmm(X, P3),y)

    loss=torch.norm(y-P4,dim=(1,2))**2 

    return loss,P4

class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma
        self.eps=1e-10

    def __call__(self, x):
        n = x.shape[0]
        x = x.view(x.shape[0], -1)
        x_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        dist= -2 * torch.mm(x, x.t()) + x_norm + x_norm.t()
        output = torch.exp(-dist / (self.sigma))

        return output


class Target(nn.Module):
    def __init__(self, embed_length=args.r, num_classes=args.nclasses_t):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 6),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(6, 6),
            nn.ReLU(),
        )
        self.classlayer = nn.Linear(6, num_classes)


    def forward(self, x1,x2):
        x=torch.cat((x1,x2),1)
        z = self.model1(x)
        out = self.classlayer(z)
        return out

class NN(nn.Module):

    def __init__(self, indim,outdim,reg,hdlayers=10):
        super().__init__()

        self.reg=reg
        self.model1 = nn.Sequential(
            nn.Linear(int(indim/2), int(hdlayers/2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(hdlayers/2),int( hdlayers/2)), 

        )
        self.model2 = nn.Sequential(
            nn.Linear(int(hdlayers/2)*2, hdlayers),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hdlayers,int( hdlayers/2)),

            nn.ReLU(),
            nn.Dropout(0.25),

        )

        self.classlayer1 = nn.Linear(int(hdlayers/2), outdim)



    def forward(self, x,y):

        out_x = self.model1(x)
        out_y = self.model1(y)
        out_z=torch.cat((out_x, out_y), 2)
        
        out_z = self.model2(out_z)
        out_z=out_z.mean(axis=1)
        out_z = self.classlayer1(out_z)

        out_x = out_x / torch.norm(out_x, dim=(1, 2)).expand(out_x.shape[2], out_x.shape[1], out_x.shape[0]).transpose(0,
                                                                                                                      2)
        out_y = out_y / torch.norm(out_y, dim=(1, 2)).expand(out_y.shape[2], out_y.shape[1], out_y.shape[0]).transpose(
            0, 2)

        # regression
        rx,px=linear_regressor(out_x,y,self.reg)
        ry,py=linear_regressor(out_y,x,self.reg)

        # concat feature and mse
        rxy= torch.cat((rx.unsqueeze(1),ry.unsqueeze(1)),1)
        out_e=torch.cat((rxy, (torch.min(rx,ry)/torch.max(rx,ry)).unsqueeze(1)), 1)

        return out_e,out_z,rx,ry
