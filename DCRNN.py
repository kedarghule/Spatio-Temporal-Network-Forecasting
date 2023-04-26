import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

DEVICE = torch.device('cuda') # cuda
shuffle=False
batch_size = 64

def get_dataset_DCRNN():
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset()

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio = 0.8)
    train_dataset, val_dataset = temporal_signal_split(train_dataset, train_ratio=0.8)

    # for batches
    #https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/a3tgcn2_example.py
    train_input = np.array(train_dataset.features) # (27399, 207, 2, 12)
    train_target = np.array(train_dataset.targets) # (27399, 207, 12)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=False, drop_last=True)

    val_input = np.array(val_dataset.features) # (27399, 207, 2, 12)
    val_target = np.array(val_dataset.targets) # (27399, 207, 12)
    val_x_tensor = torch.from_numpy(val_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_dataset_new = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset_new, batch_size=batch_size, shuffle=False,drop_last=True)

    test_input = np.array(test_dataset.features) # (, 207, 2, 12)
    test_target = np.array(test_dataset.targets) # (, 207, 12)
    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    #print(test_x_tensor.shape)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    #print(test_target_tensor.shape)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    #test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader


#https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/dcrnn_example.py
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_size):
        super(RecurrentGCN, self).__init__()
        #self.embedding = nn.Embedding(node_features, hidden_size)
        self.encode1 = DCRNN(node_features, hidden_size, 3)
        self.encode2 = DCRNN(hidden_size, hidden_size, 3)
        self.decode1 = DCRNN(node_features, hidden_size, 3)
        self.decode2 = DCRNN(hidden_size, hidden_size, 3)
        #self.dropout = torch.nn.Dropout(0.33)
        self.linear = torch.nn.Linear(hidden_size, node_features)

    # x needs to be 207 x 12
    def forward(self, x, edge_index, edge_weight):
        # x is B, N, F, T
        x = x.permute(0,1,3,2)
        # x is B, N, T, F
        #print('x_in shape: ', x_in.shape)
        # get just speed
        x = x[:,:,:,0]
        #reshape to (B*N, T)
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        #x = self.embedding(x)
        #x = F.relu(x)
        h_enc1 = self.encode1(x, edge_index, edge_weight)
        h = F.relu(h_enc1)
        h_enc2 = self.encode2(h, edge_index, edge_weight)
        h = self.decode1(x, edge_index, edge_weight, h_enc1)
        h = F.relu(h)
        h = self.decode2(h, edge_index, edge_weight, h_enc2)
        h = self.linear(h)
        return h
    

def train_dcrnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'

    model = RecurrentGCN(node_features = 12, hidden_size=64)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_decay_ratio=0.1

    #steps = [20,30,40,50,60,70,80,90]
    steps = [20, 30, 40]

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)

    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=lr_decay_ratio)

    loss_fn = torch.nn.MSELoss()
    loss_fn2 = F.l1_loss

    # Loading the graph once because it's a static graph
    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(DEVICE)
        static_edge_attr = snapshot.edge_attr.to(DEVICE)
        break

    # Training the model 
    model.train()

    epoch_vals = []
    epoch_trains = []
    for epoch in range(50):
        step = 0
        loss_list = []
        val_loss_list = []
        for encoder_inputs, labels in train_loader:
            y_hat = model(encoder_inputs, static_edge_index, static_edge_attr)         # Get model predictions
            # reshape back to BxNxT
            y_hat = y_hat.reshape((labels.shape))
            mean = [53.59967, 0.4982691]
            std = [20.209862, 0.28815305]
            labels = labels*std[0] + mean[0]
            y_hat = y_hat*std[0] + mean[0]
            loss = loss_fn2(y_hat, labels) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step= step+ 1
            #loss = torch.sqrt(loss)
            loss_list.append(loss.item())
            if step % 100 == 0 :
                print("    train MAE: ", sum(loss_list)/len(loss_list))
        lr_scheduler.step()
        train_MAE = sum(loss_list)/len(loss_list)
        epoch_trains.append(train_MAE)
        print("Epoch {} train MAE: {:.4f}".format(epoch, train_MAE))

        model.eval()
        with torch.no_grad():
            for encoder_inputs, labels in val_loader:
                y_hat = model(encoder_inputs, static_edge_index, static_edge_attr)         # Get model predictions
                # reshape back to BxNxT
                y_hat = y_hat.reshape((labels.shape))
                mean = [53.59967, 0.4982691]
                std = [20.209862, 0.28815305]
                labels = labels*std[0] + mean[0]
                y_hat = y_hat*std[0] + mean[0]
                loss = loss_fn2(y_hat, labels) 
                val_loss_list.append(loss.item())
                #if step % 100 == 0 :
                #    print("    train MAE: ", sum(loss_list)/len(loss_list))
        val_MAE = sum(val_loss_list)/len(val_loss_list)
        epoch_vals.append(val_MAE)
        print("Epoch {} val MAE: {:.4f}".format(epoch, val_MAE))