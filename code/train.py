import os
from typing import Tuple, Optional, List

import fire
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import json

from vit.architecture import VITForClassification
from vit.utils import train_one_epoch, valid_one_epoch

def save_metrics(train_losses : List[float],
                 valid_losses : List[float],
                 valid_accuracies : List[float],
                 path_to_save : str):
    os.makedirs(path_to_save, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))    
    fig.suptitle('Metrics')
    
    ax1.plot(train_losses, label='train cross entropy')
    ax1.plot(valid_losses, label='valid cross entropy')
    ax1.set_title('CrossEntropyLoss')
    ax1.legend()
    
    ax2.plot(valid_accuracies, label='valid accuracy')
    ax2.set_title('Accuracy Valid')
    ax2.legend()

    metrics_path = os.path.join(path_to_save, 'metrics.jpg')
    fig.savefig(metrics_path, bbox_inches='tight')
    
def save_to_json(path_to_save : str,
                 **kwargs):
    json_path = os.path.join(path_to_save, 'hyperparams.json')
    with open(json_path, 'w') as w:
        json.dump(kwargs, w)
    

def train(metrics_path : str,
          image_size : int = 32,
          num_classes : int = 10,
          patch_size : int = 8,
          num_channels : int = 3,
          num_hidden_layers : int = 6,
          num_attention_heads : int = 8,
          hidden_size : int = 128,
          hidden_dropout_prob : float = 0.0,
          qkv_dropout : float = 0.0,
          qkv_bias : bool = True,
          intermediate_size : int = 4*128,
          batch_size : int = 256,
          learning_rate : float = 5e-3,
          n_epochs : int = 100,
          dataset_path : str = '../data/',
          device : str = 'cuda:0'):
    vit = VITForClassification(image_size = image_size,
                               num_classes = num_classes,
                               patch_size = patch_size,
                               num_channels = num_channels,
                               num_hidden_layers = num_hidden_layers,
                               num_attention_heads = num_attention_heads,
                               hidden_size = hidden_size,
                               hidden_dropout_prob = hidden_dropout_prob,
                               qkv_dropout = qkv_dropout,
                               qkv_bias = qkv_bias,
                               intermediate_size = intermediate_size).to(device)

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)
    
    dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(vit.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    
    for _ in tqdm.tqdm(range(n_epochs)):
        train_loss = train_one_epoch(vit,
                                     optimizer,
                                     loss_fn,
                                     dl_train,
                                     device)

        valid_loss, valid_accuracy = valid_one_epoch(vit, 
                                                     loss_fn, 
                                                     dl_valid,
                                                     device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        
    print(train_losses)
    print(valid_losses)
    print(valid_accuracies)
    
    save_metrics(train_losses,
                 valid_losses,
                 valid_accuracies,
                 metrics_path)
    
    save_to_json(metrics_path,
                 best_valid_ce = np.max(valid_losses),
                 best_valid_accuracy = np.max(valid_accuracies),
                 image_size = image_size,
                 patch_size = patch_size,
                 num_classes = num_classes,
                 num_hidden_layers = num_hidden_layers,
                 num_attention_heads = num_attention_heads,
                 hidden_size = hidden_size,
                 hidden_dropout_prob = hidden_dropout_prob,
                 qkv_dropout = qkv_dropout,
                 qkv_bias = qkv_bias,
                 intermediate_size = intermediate_size,
                 batch_size = batch_size,
                 learning_rate = learning_rate,
                 n_epochs = n_epochs)
    
if __name__ == '__main__':
    fire.Fire(train)
    