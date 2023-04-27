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
import utils
import METRDataset  
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional
import torch.optim as optim
import time
from tqdm import tqdm
from gman_model import GMAN


def save_checkpoint(state, iteration,filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint",flush=True)
    torch.save(state, 'logs/'+filename+'_iteration_'+str(iteration))
    
    


def train_fn(DEVICE, loader, model, optimizer, loss_fn, nodes, D, scaler=None):
    total_train_loss=[]
    se = utils.get_spatial_embedding(nodes,D,DEVICE).to(DEVICE)
    start_time=time.time()
    count=0
    for (batch_idx, (data, targets,te)) in enumerate(loader):
        if(batch_idx%200==0):
            print(batch_idx,flush=True)
            time_update = time.time()-start_time
            start_time = time.time()
            print("time for {} batches:{}".format(count*200,time_update),flush=True)
            count+=1
            # save model
            checkpoint = {"state_dict": model.state_dict(),"optimizer":optimizer.state_dict(),}
            save_checkpoint(checkpoint,batch_idx)
        data = data.to(device=DEVICE, dtype=torch.float)
        targets = targets.to(device=DEVICE, dtype=torch.float)
        te = te.to(DEVICE)
        data = data.permute(0,2,1,3)[:,:,:,0]
        predictions = model(data,se,te)
        predictions = predictions.permute(0,2,1)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss.append(loss.item())
    return total_train_loss


def check_accuracy(DEVICE, loader, model,loss_fn,mean_stats,std_stats):
    se = utils.get_spatial_embedding(nodes,D,DEVICE).to(DEVICE)
    total_eval_loss=[]
    model.eval()
    with torch.no_grad():
        for (batch_idx, (data, targets,te)) in enumerate(loader):
            data = data.to(device=DEVICE, dtype=torch.float)
            targets = targets.to(device=DEVICE, dtype=torch.float)
            te = te.to(DEVICE)
            data = data.permute(0,2,1,3)[:,:,:,0]
            predictions = model(data,se,te)
            predictions = predictions.permute(0,2,1)
            total_eval_loss.append(loss_fn(predictions,targets).item())
    mean_eval_error = np.mean(total_eval_loss)
    unnorm_eval_error = mean_eval_error*std_stats[0].detach().numpy()+mean_stats[0].detach().numpy()
    print("Eval error: {}".format(unnorm_eval_error),flush=True)
if __name__ == "__main__":
    K=8
    d=8
    D=K*d
    L=3
    nodes=207
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    NUM_EPOCHS = 1
    gman_model = GMAN(L,K,d,12,0.9,288,True,True)
    gman_model = gman_model.to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(gman_model.parameters(), lr=LEARNING_RATE)
    total_train_loss=[]
    start_time = time.time()
    train_loss=[]
    train_loader, val_loader, test_loader,mean_stats,std_stats = utils.get_loaders()
    for i in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, gman_model, optimizer, loss_fn,nodes,D)
        total_train_loss.append(train_loss)
        print("time for epoch: {}".format(time.time()-start_time),flush=True)
    print("saving train loss list...",flush=True)
    #checkpoint = {"state_dict": gman_model.state_dict()}
    checkpoint = gman_model.state_dict()
    save_checkpoint(checkpoint,0,'final_model')
    col_length = len(total_train_loss)
    #columns = [str(i) for i in range(col_length)]
    df = pd.DataFrame(total_train_loss,index=None)
    #df.columns = columns
    df.to_csv('logs/train_loss')
    # with open('logs/train_loss.npy', 'wb') as f:
    #     np.save(f, total_train_loss,dtype=object)
    print("Evaluate model on test set",flush=True)
    eval_loss = nn.L1Loss()
    check_accuracy(test_loader, gman_model, eval_loss, mean_stats,std_stats)