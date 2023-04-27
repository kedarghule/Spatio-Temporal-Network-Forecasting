import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
import torch_geometric as tg
from torch.utils.data import DataLoader
from torch_geometric_temporal import temporal_signal_split
import torch.nn as nn
from typing import Union, Callable, Optional
import torch.optim as optim


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=64, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(64, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

