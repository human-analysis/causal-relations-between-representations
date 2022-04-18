import torch
from torch.utils.data import Dataset
import numpy as np
from .causalfunction import *
from sklearn import mixture

class Causalpairs_syn(Dataset):

    def __init__(self,ntrain,npairs,ndimsx,ndimsy,total_classes, idx,train=True):

        self.mechanism = {'linear': LinearFun,
                          'hadamard': HadamardFun,
                          'bilinear':BilinearFun,
                          'cubicspline':Cubicspline,
                          'nn':NN,
                        }
        self.ntrain=ntrain
        self.train=train
        self.npairs=npairs
        self.ndimsx=ndimsx
        self.ndimsy=ndimsy
        self.device='cuda'
        self.total_classes=total_classes
        self.idx=idx

        if not train:
            data_all = []
            label_all = []
            dataset_config=np.loadtxt('./data/val_data_%d'%(self.idx) + '/config.csv', delimiter=',')
            num_data=dataset_config.shape[0]
            for pair in range(0,num_data):
                data = np.loadtxt('./data/val_data_%d'%(self.idx) + '/%05d.csv' % (pair), delimiter=',')#m1_depvalue
                X = data[:, 0:-1]
                Y = data[:, -1]
                Y[Y == -1] = 2
                Y = Y[0]
                data_all.append(X)
                label_all.append(Y)
            data_all = np.array(data_all)
            label_all = np.array(label_all)

            self.X = torch.from_numpy(data_all[:,:,0:ndimsx]).float()
            self.Y = torch.from_numpy(data_all[:,:,ndimsx:ndimsx+ndimsy]).float()
            self.label = torch.from_numpy(label_all)

    def scale_tensor(self,data):
        mean=data.mean(0)
        std=data.std(0)
    
        return (data-mean)/(std+1e-5)

    def initial_data(self,nsamples,ndims):
        g  = mixture.GaussianMixture(n_components=ndims,covariance_type='spherical')
        k=3
        p1=4
        p2=1
        g.means_ = p1 * np.random.randn(k, ndims)
        g.covariances_ = np.power(abs(p2 * np.random.randn(k, ndims) + 1), 2)
        g.weights_ = abs(np.random.rand(k))
        g.weights_ = g.weights_ / sum(g.weights_)
        data=g.sample(nsamples)
        data=data[0]
        outputs=torch.tensor(data,dtype=torch.float32)
        
        return self.scale_tensor(outputs)

    def getdata(self,npairs,ndimsx,ndimsy):
 
        p_config = np.random.randint(0, 6)  # 6 different causal cases


        p_fun = self.idx
        while p_fun==self.idx:
            p_fun = np.random.randint(0, 5)

        funs=['linear','hadamard','bilinear','cubicspline','nn']
        fun=funs[p_fun]

        if p_config ==0: # X->Y
            X=self.initial_data(npairs,ndimsx)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsx)
            X=causal_mechanism(X)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsy)
            Y=causal_mechanism(X)
            label = 1
        elif p_config ==1: # Y->X
            Y=self.initial_data(npairs,ndimsy)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsy)
            Y=causal_mechanism(Y)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsx)
            X=causal_mechanism(Y)
            label = 2
        elif p_config ==2: #X Y
            X=self.initial_data(npairs,ndimsx)
            Y=self.initial_data(npairs,ndimsy)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsx)
            X=causal_mechanism(X)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsy)
            Y=causal_mechanism(Y)
            label=0
        elif p_config ==3: #X<-Z->Y
            ndimsz=min(ndimsx,ndimsy)
            Z=self.initial_data(npairs,ndimsz)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsz)
            Z=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
            X=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
            Y=causal_mechanism(Z)
            label = 0
        elif p_config ==4: #X<-Z->Y and X->Y
            ndimsz=min(ndimsx,ndimsy)
            Z=self.initial_data(npairs,ndimsz)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsz)
            Z=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
            X=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsy,ndimsz)
            Y=causal_mechanism(X,Z)
            label = 1
        elif p_config == 5:  # X<-Z->Y and Y->X
            ndimsz=min(ndimsx,ndimsy)
            Z=self.initial_data(npairs,ndimsz)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsz)
            Z=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
            Y=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsx,ndimsz)
            X=causal_mechanism(Y,Z)
            label = 2
        domain_label=p_fun

        return X,Y,label,domain_label

    def __len__(self):

        if self.train:
            l=self.ntrain
        else:
            l=len(self.X)
        
        return l

    def __getitem__(self, index):

        if self.train:
            X,Y,label,sensitives_label=self.getdata(self.npairs,self.ndimsx,self.ndimsy)
            if self.idx!=self.total_classes:
                if sensitives_label==self.total_classes:
                    sensitives_label=self.idx
            return X,Y,label,sensitives_label
        else:
            X,Y,label=self.X[index],self.Y[index],self.label[index]
            sensitives_label=self.idx

            return X,Y,label,sensitives_label