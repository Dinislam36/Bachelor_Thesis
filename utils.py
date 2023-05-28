import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torch import nn
from torch_geometric.nn import GCNConv

from torch_geometric_temporal.nn import GConvLSTM


from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import plotly.express as px


def split_dataset(data, train_size=0.8):
    train_len = int(len(data) * train_size)
    data_train = data[:train_len]
    data_test = data[train_len:]
    
    return data_train, data_test


def get_edge_index(distances, area2idx):
    distances['from'] = distances['from'].apply(lambda x: area2idx[x])
    distances['to'] = distances['to'].apply(lambda x: area2idx[x])
    edge_idx = distances[['from', 'to']].to_numpy()
    return edge_idx.transpose()


def train_step(model, X, y, edge_idx, edge_weight, optimizer, loss_fn):
    out = model(X, edge_idx, edge_weight)
    loss = loss_fn(out, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return out, loss.item()

def test_step(model, X, y, edge_idx, edge_weight, loss_fn):
    out = model(X, edge_idx, edge_weight)
    return out, loss_fn(out, y).item()

class CovidCasesDataset():
    def __init__(self, dataset, train_window, test_window):
        self.dataset = dataset
        self.train_window = train_window
        self.test_window = test_window
        self.len = dataset.shape[0] - train_window - test_window
        assert self.len >= 0, "Small dataset size or large windows"
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.dataset[idx:idx + self.train_window], self.dataset[idx + self.train_window:idx + self.train_window + self.test_window]
    

