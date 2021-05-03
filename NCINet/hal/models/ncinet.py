import torch.nn as nn
import torch
import config
import numpy as np

args = config.parse_args()

__all__ = ['NCINet']

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


class MetricRenyi(nn.Module):
    def __init__(self, normalize=True, alpha=2, sigma=1):
        super(MetricRenyi, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.normalize = normalize
        self.kernel = GaussianKernel(sigma=self.sigma)
        self.eps=1e-10

    def renyi_entropy(self, x):
        k = self.kernel(x)
        if torch.trace(k)==0:
            import pdb;pdb.set_trace()
        k = k / (torch.trace(k))
        eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eig_pow = eigv ** self.alpha
        # if torch.log2(torch.sum(eig_pow))==0:
        #     import pdb;pdb.set_trace()
        entropy = (1 / (1 - self.alpha)) * torch.log2(torch.sum(eig_pow))
        return entropy

    def joint_entropy(self, x, y):
        x = self.kernel(x)
        y = self.kernel(y)
        k = torch.mul(x, y)
        # if torch.trace(k)==0:
        #     import pdb;pdb.set_trace()
        k = k / (torch.trace(k))
        eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        # if torch.sum(eigv)==0:
        #     import pdb;pdb.set_trace()
        eig_pow =  eigv ** self.alpha
        entropy = (1 / (1 - self.alpha)) * torch.log2(torch.sum(eig_pow))
        return entropy

    def forward(self, inputs, target):
        Hx = self.renyi_entropy(inputs)
        Hy = self.renyi_entropy(target)
        Hxy = self.joint_entropy(inputs, target)
        
        if self.normalize:
            Ixy = Hx + Hy - Hxy
            # if torch.max(Hx, Hy)==0:
            #     import pdb;pdb.set_trace()
            Ixy = Ixy / (torch.max(Hx, Hy))
        else:
            Ixy = Hx + Hy - Hxy
        
        return Ixy

def linear_regressor(x,y,reg):
    eps=1e-10
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

class NCINet(nn.Module):

    def __init__(self, indim,outdim,reg,hdlayers=10):
        super().__init__()

        self.reg=reg

        self.model11 = nn.Sequential(
            nn.Linear(int(indim/2), int(hdlayers/2)),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(int(hdlayers/2),int( hdlayers/2)), 

        )
        self.model12 = nn.Sequential(
            nn.Linear(int(hdlayers/2)*2, hdlayers),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hdlayers,int( hdlayers/2)),

            nn.ReLU(),
            # nn.Dropout(),

        )
       
        # classification 2
        self.model2 = nn.Sequential(
            nn.Linear(outdim+2, int(hdlayers/2)),
            nn.ReLU(),
            # nn.Dropout(),

        )

        self.classlayer1 = nn.Linear(int(hdlayers/2), outdim)
        self.batchnorm = nn.BatchNorm1d(num_features=outdim+2)
        self.softmaxlayer1 = nn.Softmax(dim=1)

        self.classlayer2 = nn.Linear(int(hdlayers/2), outdim)
        self.softmaxlayer2 = nn.Softmax(dim=1)



    def forward(self, x,y):


        out_x = self.model11(x)
        out_y = self.model11(y)
        out_c=torch.cat((out_x, out_y), 2)
        
        out_c = self.model12(out_c)
        out_c=out_c.mean(axis=1)
        out_c = self.classlayer1(out_c)
        #
        #out_c = out_c/torch.norm(out_c,dim=1).expand(3,10).transpose(0,1)
        prob1=self.softmaxlayer1(out_c)

        # regression
        out_x=out_x/torch.norm(out_x, dim=(1,2)).expand(out_x.shape[2],out_x.shape[1],out_x.shape[0]).transpose(0,2)
        out_y=out_y/torch.norm(out_y, dim=(1,2)).expand(out_y.shape[2],out_y.shape[1],out_y.shape[0]).transpose(0,2)
        xx=x/torch.norm(x, dim=(1,2)).expand(x.shape[2],x.shape[1],x.shape[0]).transpose(0,2)
        yy=y/torch.norm(y, dim=(1,2)).expand(y.shape[2],y.shape[1],y.shape[0]).transpose(0,2)
        
        # out_x=out_x/torch.norm(out_x, dim=(1,2))
        # out_y=out_y/torch.norm(out_y, dim=(1,2))
        rx,px=linear_regressor(out_x,yy,self.reg)
        ry,py=linear_regressor(out_y,xx,self.reg)
        
        ex=torch.zeros((px.shape[0])).to('cuda')
        ey=torch.zeros((py.shape[0])).to('cuda')
        renyi = MetricRenyi()
        for i in range(px.shape[0]):
            ex[i]=MetricRenyi()(yy[i]-px[i], x[i])#yy[i]-
            ey[i]=MetricRenyi()(xx[i]-py[i], y[i])#xx[i]-

        
        # concat logits and mse
        rxy= torch.cat((ex.unsqueeze(1),ey.unsqueeze(1)),1)
        in_c=torch.cat((out_c, rxy), 1)
        in_c=self.batchnorm(in_c)

        # classification 2
        in_c=self.model2(in_c)
        out_c_f=self.classlayer2(in_c)

        prob2 = self.softmaxlayer2(out_c_f)

        return out_c,rx,ry,ex,ey,out_c_f,prob1,prob2 