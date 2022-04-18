import torch
import os
import numpy as np
from causalfunction import *
from sklearn import mixture

class PrepareCAUSAL():

    def __init__(self,npairs,ndimsx,ndimsy,idx):

        self.mechanism = {'linear': LinearFun,
                          'hadamard':HadamardFun,
                          'bilinear':BilinearFun,
                          'cubicspline':Cubicspline,
                          'nn':NN,
                        }
        self.npairs=npairs
        self.ndimsx=ndimsx
        self.ndimsy=ndimsy
        self.device="cuda:0"
        self.idx=idx
         
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

        p_config = np.random.randint(0, 5)  # 6 different causal cases
        p_fun=self.idx
        print(p_fun)      
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

        return X,Y,label,p_fun,p_config

def main():
    for idx in range(5):  # 5 different causal functions

        num_test = 600
        npairs = 100  # numbers of pairs per sample
        ndimsx = 8  # dimensionality x
        ndimsy = 8  # dimensionality y
        generator = PrepareCAUSAL(npairs=npairs, ndimsx=ndimsx, ndimsy=ndimsy, idx=idx)
        savepath = './val_data_' + str(idx) + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        X_all = torch.zeros((num_test, npairs, ndimsx), dtype=torch.float)
        Y_all = torch.zeros((num_test, npairs, ndimsy), dtype=torch.float)
        label_all = torch.zeros(num_test)

        ps = np.zeros((num_test, 2))
        for pair in range(num_test):
            X, Y, label, pf, pc = generator.getdata(npairs, ndimsx, ndimsy)
            X_all[pair] = X
            Y_all[pair] = Y
            label_all[pair] = label
            ps[pair, 0] = pf
            ps[pair, 1] = pc
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()
            data = np.hstack((X, Y))
            data = np.hstack((data, np.ones((npairs, 1)) * label))
            np.savetxt(savepath + '%05d.csv' % (pair), data, delimiter=',')

        np.savetxt(savepath + 'config.csv', ps, delimiter=',')

if __name__ == "__main__":
    main()