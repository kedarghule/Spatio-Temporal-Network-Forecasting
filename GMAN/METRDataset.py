import torch
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from torch.utils.data import DataLoader
from torch_geometric_temporal import temporal_signal_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric_temporal.dataset import METRLADatasetLoader


class METRDataset(Dataset):
    
    def get_temporal_embedding(self,x,y, pred_steps):
        x = torch.mean(x,dim=0)
        x = x[:,1]
        #print(x.shape)
        y_temp=[]
        cd = x[-1] - x[-2]
        a = x[-1]
        for j in range(pred_steps):
            y_temp.append(a+(j+1)*cd)
        y_temporal = torch.tensor(y_temp)
        y_temporal = y_temporal.to(x.device)
        #print(y_temporal.shape)

        te = torch.concat((x,y_temporal),dim=0)
        #print(te.shape)
        return te
    
    
    
    def renormalize_data(self, X, y, mean=None,std=None, train_flag=False):
        if(train_flag):
            if(mean==None or std==None):
                raise AttributeError('mean or std not passed to train Dataset')
            # unnormalize data
            X = X*std + mean
            y = y*std[0][0] + mean[0][0]
            # get normalization constants from training data
            X_mean = torch.mean(X,(0,1,2))
            X_std = torch.std(X,(0,1,2))
            y_mean = torch.mean(y,(0,1,2))
            y_std = torch.std(y,(0,1,2))
            # normalize data using the obtained constants
            X = (X-X_mean)/X_std
            y = (y-y_mean)/y_std
            return X,y,X_mean,X_std,y_mean,y_std
        else:
            assert(mean==None and std==None)
            # unnormalize data
            X = X*self.global_std + self.global_mean
            y = y*self.global_std[0][0] + self.global_mean[0][0]
            # normalize data using the obtained constants
            X = (X-self.X_mean)/self.X_std
            y = (y-self.y_mean)/self.y_std
        return X,y

    def __init__(self, X, y, train_flag=False, mean=None, std = None, trainMean_X=None, trainMean_y = None , trainStd_X=None
                 ,trainStd_y=None):
        if(train_flag):
            assert(trainMean_X is None and trainStd_X is None)
            assert(trainMean_y is None and trainStd_y is None)
            self.global_mean = mean
            self.global_std = std
            X,y,self.X_mean,self.X_std,self.y_mean,self.y_std = self.renormalize_data(X,y,mean,std,True)
        else:
            self.global_mean = mean
            self.global_std = std
            self.X_mean,self.X_std,self.y_mean,self.y_std = trainMean_X, trainStd_X, trainMean_y, trainStd_y
            X,y = self.renormalize_data(X,y,None, None, False)
        
        self.X=X
        self.y=y
        
    def __len__(self):
        assert(self.X.shape[0]==self.y.shape[0])
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        #print(x.shape)
        te = self.get_temporal_embedding(x,y,x.shape[1])
        return x,y, te
    
    


