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
import METRDataset


def get_data():
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset()
    z_score_mean = dataset.meanValues
    z_score_std = dataset.stdValues
    A = np.load('data/adj_mat.npy')
    train_val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    train_dataset, val_dataset = temporal_signal_split(train_val_dataset, train_ratio = 0.8)
    
    X_train = torch.Tensor(np.array(train_dataset.features))
    X_train = X_train.permute(0, 1, 3, 2)
    y_train = torch.Tensor(np.array(train_dataset.targets))

    X_val = torch.Tensor(np.array(val_dataset.features))
    X_val = X_val.permute(0, 1, 3, 2)
    y_val = torch.Tensor(np.array(val_dataset.targets))

    X_test = torch.Tensor(np.array(test_dataset.features))
    X_test = X_test.permute(0, 1, 3, 2)
    y_test = torch.Tensor(np.array(test_dataset.targets))
    
    mean_stats = torch.tensor(z_score_mean)
    mean_stats = torch.unsqueeze(mean_stats,1)
    mean_stats = torch.permute(mean_stats,[1,0])
    
    std_stats = torch.tensor(z_score_std)
    std_stats = torch.unsqueeze(std_stats,1)
    std_stats = torch.permute(std_stats,[1,0])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, mean_stats, std_stats


def get_loaders(batch_size=16):
    X_train, y_train, X_val, y_val, X_test, y_test, data_means, data_stds = get_data()
    train_dataset = METRDataset.METRDataset(X_train, y_train, True, data_means,data_stds)
    val_dataset = METRDataset.METRDataset(X_val, y_val, False, train_dataset.global_mean, train_dataset.global_std, 
                         train_dataset.X_mean, train_dataset.y_mean, train_dataset.X_std, train_dataset.y_std)
    test_dataset = METRDataset.METRDataset(X_test, y_test, False, train_dataset.global_mean, train_dataset.global_std, 
                         train_dataset.X_mean, train_dataset.y_mean, train_dataset.X_std, train_dataset.y_std)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
    val_datalaoder = DataLoader(val_dataset, batch_size=16, shuffle=False,drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False,drop_last=True)
    return train_dataloader, val_datalaoder, test_dataloader, data_means, data_stds


def get_spatial_embedding(nodes, D, device):
        seq_len = nodes
        input_dim = D
        pe = torch.zeros(seq_len,1,input_dim).to(device)
        den  = 1/seq_len**((1/input_dim)*torch.arange(0,input_dim,2))
        num = torch.arange(seq_len).reshape(seq_len,1)
        pe[:,0,0::2] = torch.sin(num*den)
        pe[:,0,1::2] = torch.cos(num*den)
        pe=pe.squeeze()
        return pe