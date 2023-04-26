import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import math
import typing
import torch
import random
import matplotlib.pyplot as plt
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention.stgcn import TemporalConv

torch.manual_seed(200)
torch.cuda.manual_seed(200)

class STGCNBlock(nn.Module):
    """
    ST-Conv Block of the STGCN.
    Temporal Gated Conv->Spatial Graph Conv->ReLU->Temporal Gated-Conv->Layer Norm
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, block_no):
        """
        Input:
        in_channels: Number of input features at each node in each time step.
        spatial_channels: Number of output channels of the graph convolutional, spatial sub-block.
        out_channels: Desired number of output features at each node in each time step.
        num_nodes: Number of nodes in the graph.
        block_no: STGCN Block number
        """
        super(STGCNBlock, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.layer_norm1 = nn.LayerNorm([num_nodes, 8, out_channels])
        self.layer_norm2 = nn.LayerNorm([num_nodes, 4, out_channels])
        self.temporal_conv1 = TemporalConv(in_channels=in_channels, out_channels=out_channels)
        self.temporal_conv2 = TemporalConv(in_channels=spatial_channels, out_channels=out_channels)
        self.block_no = block_no
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        Input:
        X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features=in_channels).
        A_hat: Normalized adjacency matrix.
        Output data of shape (batch_size, num_nodes, num_timesteps_out, num_features=out_channels).
        """
        X = X.permute(0, 2, 1, 3)
        t = self.temporal_conv1(X)
        t = t.permute(0, 2, 1, 3)
        
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))

        t2 = t2.permute(0, 2, 1, 3)
        t3 = self.temporal_conv2(t2)
        t3 = t3.permute(0, 2, 1, 3)
        
        if self.block_no == 1:
            t4 = self.layer_norm1(t3)
        elif self.block_no == 2:
            t4 = self.layer_norm2(t3)
        return t4


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network by Yu et al.
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        Input:
        num_nodes: Number of nodes in the graph.
        num_features: Number of features at each node in each time step.
        num_timesteps_input: Number of past time steps fed into the network.
        num_timesteps_output: Desired number of future time steps output by the network.
        """
        super(STGCN, self).__init__()
        self.stgcn_block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, block_no=1)
        self.stgcn_block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, block_no=2)
        self.temporal_layer = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 1))
        self.fcn = nn.Linear(64, num_timesteps_output)


    def forward(self, A_hat, X):
        """
        Input:
        X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features=in_channels).
        A_hat: Normalized adjacency matrix.
        """
        out1 = self.stgcn_block1(X, A_hat)
        out2 = self.stgcn_block2(out1, A_hat) # (50, 207, 4, 64)
        out2 = out2.permute(0, 3, 2, 1)
        out3 = self.temporal_layer(out2) # 50, 64, 1, 207
        out3 = out3.permute(0, 3, 1, 2)
        out4 = self.fcn(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4
