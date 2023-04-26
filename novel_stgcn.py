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
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, block_no):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.temporal_conv1 = TemporalConv(in_channels=in_channels, out_channels=out_channels)
        self.temporal_conv2 = TemporalConv(in_channels=spatial_channels, out_channels=out_channels)
        self.block_no = block_no
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        X = X.permute(0, 2, 1, 3)
        t = self.temporal_conv1(X)
        t = t.permute(0, 2, 1, 3)
        
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))

        t2 = t2.permute(0, 2, 1, 3)
        t3 = self.temporal_conv2(t2)
        t3 = t3.permute(0, 2, 1, 3)
        
        #print("Shape before norm = ", t3.size())
        t4 = self.batch_norm(t3)
        #t4 = self.dropout(t4)
        return t4


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.stgcn_block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, block_no=1)
        self.stgcn_block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, block_no=2)
        self.extra_temporal_conv = TemporalConv(in_channels=64, out_channels=64) # Added extra layer
        self.fcn = nn.Linear((num_timesteps_input - 2 * 5) * 64, num_timesteps_output) #Changed this

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.stgcn_block1(X, A_hat)
        out2 = self.stgcn_block2(out1, A_hat)
        
        #print("Shape going inside extra temporal layer = ", out2.size())
        out2 = out2.permute(0, 2, 1, 3)
        out3 = self.extra_temporal_conv(out2)
        out3 = out3.permute(0, 2, 1, 3)
        #print("Shape coming outside extra temporal layer = ", out3.size())
        
        out4 = self.fcn(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4
