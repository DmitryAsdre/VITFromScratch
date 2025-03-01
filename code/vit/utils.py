from typing import Tuple

import numpy as np

import torch
import torch.nn as nn


def train_one_epoch(model : nn.Module,
                    optimizer : torch.optim.Optimizer,
                    loss_fn : torch.nn.CrossEntropyLoss,
                    dl_train : torch.utils.data.DataLoader,
                    device : str = 'cuda:0') -> float:
    model = model.train()
    
    epoch_losses = []
    for imgs, labels in dl_train:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(imgs)
        ce_loss = loss_fn(logits, labels)
        
        ce_loss.backward()
        epoch_losses.append(ce_loss.item())
        optimizer.step()
    
    return np.mean(epoch_losses)

def valid_one_epoch(model : nn.Module,
                    loss_fn : torch.nn.CrossEntropyLoss,
                    dl_valid : torch.utils.data.DataLoader,
                    device : str = 'cuda:0') -> Tuple[float, float]:
    model = model.eval()
    
    ce_losses = []
    acc_losses = []
    
    with torch.no_grad():
        for imgs, labels in dl_valid:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            logits, _ = model(imgs)
            ce_loss = loss_fn(logits, labels)
            ce_losses.append(ce_loss.item())
            
            labels_pred = logits.argmax(axis=1)
            accuracy = (labels_pred == labels).float().mean()
            acc_losses.append(accuracy.item())
            
    return np.mean(ce_losses),\
           np.mean(acc_losses)