import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
import time
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

# class for simple LSTM

class TrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size:  int, num_layers: int, sequence_len: int):
        super().__init__()
        self.sequence_len = sequence_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = 0.1)
        # self.linear1 = nn.Linear(hidden_size*sequence_len, int(sequence_len*input_size/2))
        # self.relu1 = nn.ReLU()
        # self.linear2 = nn.Linear(int(sequence_len*input_size/2), sequence_len*input_size)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        #batch_size = x.shape[0]
        x, _ = self.lstm(x)
        #x = x.reshape(batch_size, -1)
        # #print("LSTM output: ", x.shape)
        # x = self.linear1(x)
        # #print("Linear output: ", x.shape)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = x.reshape(batch_size, self.sequence_len, self.input_size)
        #x = x[:, -1, :]
        x = self.linear(x)
        return x

def train(model, dataloader, loss_func, loss_func_2, device, optimizer):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    total_mae = 0
    count = 0
    for idx, (data1, label) in enumerate(dataloader):
        count += 1
        #label = label[:,-1,:]
        label = label.to(device)
        data1 = data1.to(device)
        optimizer.zero_grad()
        
        out = None
        ###########################################################################
        # TODO: compute the logits of the input, get the loss, and do the         #
        # gradient backpropagation.
        ###########################################################################
        # if(idx == 0):
        #    print("input shape: ", data1.shape)
        #    print("label shape: ", label.shape)
        out = model(data1)
        out = out.swapaxes(1,2)
        label = label.swapaxes(1,2)
        
        loss = loss_func(out, label)
        mean = [53.59967, 0.4982691]
        std = [20.209862, 0.28815305]
        # X = X - means.reshape(1, -1, 1)
        # stds = np.std(X, axis=(0, 2))
        # X = X / stds.reshape(1, -1, 1)
        label = label*std[0] + mean[0]
        out = out*std[0] + mean[0]
        # if(idx == 0):
        #     print("output shape, ", out.shape)
        #     print("label shape: ", label.shape)
        #     print("out: ", out[0][0])
        #     print("label: ", label[0][0])
        mae = loss_func_2(out, label).item()
        total_mae += mae
        loss.backward()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
        optimizer.step()

        
        train_rmse = torch.sqrt(loss)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| err {:8.3f}'.format(epoch, idx, len(dataloader),
                                              train_rmse))
            total_acc, total_count = 0, 0
            start_time = time.time()
    #print('Total MAE: ', total_mae)
    #print('Total count: ', count)
    return total_mae/count

def evaluate(model, dataloader, loss_func, loss_func_2, device):
    model.eval()
    total_acc, total_count = 0, 0
    
    predictions = []
    labels = []
    total_val_rmse = 0
    total_mae = 0
    count = 0
    with torch.no_grad():
        for idx, (data1, label) in enumerate(dataloader):
            count+=1
            label = label.to(device)
            data1 = data1.to(device)
            # if(idx == 0):
            #     print("input shape: ", data1.shape)
            #     print("label shape: ", label.shape)

            label = label.swapaxes(1,2)
            # undo z-score
            mean = [53.59967, 0.4982691]
            std = [20.209862, 0.28815305]
            # X = X - means.reshape(1, -1, 1)
            # stds = np.std(X, axis=(0, 2))
            # X = X / stds.reshape(1, -1, 1)
            label = label*std[0] + mean[0]
            
            ###########################################################################
            # TODO: compute the logits of the input, get the loss.                    #
            ###########################################################################
            logits = model(data1)
            logits = logits.swapaxes(1,2)
            logits = logits*std[0] + mean[0]
            
            #print(logits.shape)
            #print(label.shape)
            # if(idx == 0):
            #     print("output shape, ", logits.shape)
            #     print("label shape: ", label.shape)
            #     print("out: ", logits[0][0])
            #     print("label: ", label[0][0])
            loss = loss_func(logits, label)
            mae = loss_func_2(logits, label).item()
            ###########################################################################
            #                             END OF YOUR CODE                            #
            ###########################################################################
            val_rmse = torch.sqrt(loss)
            #print("Validation rmse: ", val_rmse)
            #print("Validation mae: ", mae)
            predictions.append(logits.cpu())
            labels.append(label.cpu())
            #predictions.append(logits.item())
            #labels.append(label.item())
            total_val_rmse += val_rmse
            total_mae += mae
    #print('Total MAE: ', total_mae)
    #print('Total count: ', count)
    return predictions, labels, total_val_rmse/count, total_mae/count

