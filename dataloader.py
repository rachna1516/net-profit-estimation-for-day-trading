import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import numpy as np
import pandas as pd

def lstm_create_sequences(data, seq_length):
    xs, ys = list(), list()
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 0] # open price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    print(df)
    X, y = lstm_create_sequences(df.values, 1)
    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    exit()
    for x,y in dataloader:
        print(x)
        print(y)