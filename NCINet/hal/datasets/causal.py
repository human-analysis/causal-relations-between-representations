import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import math
from random import shuffle
import numpy as np
from sklearn.preprocessing import minmax_scale
import random
from .causalfunction import *
import pytorch_lightning as pl

__all__ = ['CAUSAL']

class PrepareCAUSAL(pl.LightningDataModule):

    def __init__(self, root,npairs,ndimsx,ndimsy,nranges,idx,train=True):

        self.mechanism = {'linear': LinearFun,
                          'quadratic': QuadraticFun,
                          'hadamard':HadamardFun,
                          'bilinear':BilinearFun,
                          'GP':GPFun,
                          'cubicspline':Cubicspline,
                          'nn':NN,
                        }
        self.train=train
        self.npairs=npairs
        self.ndimsx=ndimsx
        self.ndimsy=ndimsy
        self.nranges=nranges
        self.device="cuda:0"
        self.idx=idx
        if train:
            pass
        else: # testing stage, loading dataset
            data_all = []
            label_all = []
            for pair in range(0,600):
                data = np.loadtxt(root+'val_data_'+str(self.idx) + '/%05d.csv' % (pair), delimiter=',')#m1_depvalue
                X = data[0:100, 0:-1]
                Y = data[0:100, -1]
                Y[Y == -1] = 2
                Y = Y[0]
                data_all.append(X)
                label_all.append(Y)
            data_all = np.array(data_all)
            label_all = np.array(label_all)

            self.X =  torch.from_numpy(data_all[:,:,0:8]).to(torch.float)
            self.Y = torch.from_numpy(data_all[:,:,8:16]).to(torch.float)
            self.label = torch.from_numpy(label_all).to(torch.int64)
         
    def scale_tensor(self,data):
        mean=data.mean()
        std=data.std()

        return (data-mean)/std

    def initial_data(self,nsamples,ndims,nranges):
        #import pdb;pdb.set_trace()
        output=torch.rand(1)*torch.randn(nsamples,ndims)+ torch.tensor(random.sample([nranges, -nranges], 1))
        #output=np.random.rand(1) * np.random.randn(nsamples, ndims)+ random.sample([2, -2], 1)
        return self.scale_tensor(output)

    def getdata(self,npairs,ndimsx,ndimsy,nranges):

        p_config = np.random.randint(0, 6)  # 6 different causal cases
        if self.train:
            p_fun=self.idx
            while p_fun==self.idx: # making sure the training function does not include testing function
                p_fun = np.random.randint(0, 6)  # 6 different causal mechanism
        else:
            p_fun = self.idx
        funs=['linear','quadratic','hadamard','bilinear','cubicspline','nn']
        fun=funs[p_fun]
        if p_config ==0: # X->Y
            X=self.initial_data(npairs,ndimsx,nranges)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsx)
            X=causal_mechanism(X)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsy)
            Y=causal_mechanism(X)
            label = 1
        elif p_config ==1: # Y->X
            Y=self.initial_data(npairs,ndimsy,nranges)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsy)
            Y=causal_mechanism(Y)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsx)
            X=causal_mechanism(Y)
            label = 2
        elif p_config ==2: #X Y
            X=self.initial_data(npairs,ndimsx,nranges)
            Y=self.initial_data(npairs,ndimsy,nranges)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsx)
            X=causal_mechanism(X)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsy)
            Y=causal_mechanism(Y)
            label=0
        elif p_config ==3: #X<-Z->Y
            ndimsz=min(ndimsx,ndimsy)
            Z=self.initial_data(npairs,ndimsz,nranges)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsz)
            Z=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
            X=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
            Y=causal_mechanism(Z)
            label = 0
        elif p_config ==4: #X<-Z->Y and X->Y
            ndimsz=min(ndimsx,ndimsy)
            Z=self.initial_data(npairs,ndimsz,nranges)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsz)
            Z=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
            X=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
            Y=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsx,ndimsy)
            Y=causal_mechanism(X)
            label = 1
        elif p_config == 5:  # X<-Z->Y and Y->X
            ndimsz=min(ndimsx,ndimsy)
            Z=self.initial_data(npairs,ndimsz,nranges)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsz)
            Z=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
            X=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
            Y=causal_mechanism(Z)
            causal_mechanism=self.mechanism[fun](ndimsy,ndimsx)
            X=causal_mechanism(Y)
            label = 2
        # if p_config ==0: # X->Y
        #     X=self.initial_data(npairs,ndimsx,nranges)
        #     causal_mechanism=self.mechanism[fun](ndimsx,ndimsy)
        #     Y=causal_mechanism(X)
        #     label = 1
        # elif p_config ==1: # Y->X
        #     Y=self.initial_data(npairs,ndimsy,nranges)
        #     causal_mechanism=self.mechanism[fun](ndimsy,ndimsx)
        #     X=causal_mechanism(Y)
        #     label = 2
        # elif p_config ==2: #X Y
        #     X=self.initial_data(npairs,ndimsx,nranges)
        #     Y=self.initial_data(npairs,ndimsy,nranges)
        #     label=0
        # elif p_config ==3: #X<-Z->Y
        #     ndimsz=min(ndimsx,ndimsy)
        #     Z=self.initial_data(npairs,ndimsz,nranges)
        #     causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
        #     X=causal_mechanism(Z)
        #     causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
        #     Y=causal_mechanism(Z)
        #     label = 0
        # elif p_config ==4: #X<-Z->Y and X->Y
        #     ndimsz=min(ndimsx,ndimsy)
        #     Z=self.initial_data(npairs,ndimsz,nranges)
        #     causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
        #     X=causal_mechanism(Z)
        #     causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
        #     Y=causal_mechanism(Z)
        #     causal_mechanism=self.mechanism[fun](ndimsx,ndimsy)
        #     Y=causal_mechanism(X)
        #     label = 1
        # elif p_config == 5:  # X<-Z->Y and Y->X
        #     ndimsz=min(ndimsx,ndimsy)
        #     Z=self.initial_data(npairs,ndimsz,nranges)
        #     causal_mechanism=self.mechanism[fun](ndimsz,ndimsx)
        #     X=causal_mechanism(Z)
        #     causal_mechanism=self.mechanism[fun](ndimsz,ndimsy)
        #     Y=causal_mechanism(Z)
        #     causal_mechanism=self.mechanism[fun](ndimsy,ndimsx)
        #     X=causal_mechanism(Y)
        #     label = 2

        return X,Y,label,p_fun,p_config


    def __len__(self):

        if self.train:
            l=1000
        else:
            l=len(self.X)
        
        return l

    def __getitem__(self, index):

        if self.train:
            X,Y,label,p_fun,p_config=self.getdata(self.npairs,self.ndimsx,self.ndimsy,self.nranges)
        else:
            X,Y,label=self.X[index],self.Y[index],self.label[index]

        return X,Y,label

class CAUSAL(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        dataset = PrepareCAUSAL(self.opts.dataroot,self.opts.npairs,self.opts.ndimsx,self.opts.ndimsy,self.opts.nranges,self.opts.idx,train=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        dataset = PrepareCAUSAL(self.opts.dataroot,self.opts.npairs,self.opts.ndimsx,self.opts.ndimsy,self.opts.nranges,self.opts.idx,train=False)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        dataset = PrepareCAUSAL(self.opts.dataroot,self.opts.npairs,self.opts.ndimsx,self.opts.ndimsy,self.opts.nranges,self.opts.idx,train=False)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader
