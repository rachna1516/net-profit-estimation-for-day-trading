import yfinance as yf
from dataloader import StockDataset
from model import Gatedmodel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def lstm_create_sequences(data, seq_length):
    xs, ys = list(), list()
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 0] # open date
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs         = torch.tensor(X_test, dtype=torch.float32)
        targets        = torch.tensor(y_test, dtype=torch.float32)
        predictions    = model(inputs)
        predictions_np = predictions.numpy()
        targets_np     = targets.numpy()
        mse            = mean_squared_error(targets_np, predictions_np)
        mae            = mean_absolute_error(targets_np, predictions_np)
        
        return predictions_np, targets_np

def plot_predictions(predictions, targets, savefile):
    plt.figure(figsize=(14, 7))
    plt.plot(targets[:], label='Actual', color='blue')
    plt.plot(predictions[:], label='Predicted', color='red')
    plt.title('Stock Price Prediction vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig(savefile)

if __name__ == "__main__":
    # downloading the latest data
    # df = yf.download("WMT", interval="60m")  # interval 60 minutes
    # df.to_csv('data.csv')

    # loading the data
    df = pd.read_csv('data.csv')
    
    # preprocessing features and normalization
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    df_scaled = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], index=df.index)
    
    # creating sequences to preserve the temporal order of the data using sliding window
    X, y             = lstm_create_sequences(df_scaled.values, 10)
    X_trian, y_train = X[:int(df_scaled.shape[0] * 0.8)], y[:int(df_scaled.shape[0] * 0.8)]
    X_test, y_test   = X[int(df_scaled.shape[0] * 0.8):], y[int(df_scaled.shape[0] * 0.8):]
    
    # dataloader
    dataset    = StockDataset(X_trian, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # LSTM model
    model     = Gatedmodel(X.shape[2],64,32,1)
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs    = 50
    
    # training the model
    for epoch in tqdm(range(epochs)):
        train_model(model, dataloader, criterion, optimizer)
    
    # evaluation of the model
    predictions_np, targets_np = evaluate_model(model, X_test, y_test)
    mae = mean_absolute_error(targets_np, predictions_np)
    r2 = r2_score(targets_np, predictions_np)
    rmse = np.sqrt(mean_squared_error(targets_np, predictions_np))

    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R-squared Score: {r2:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')

    # plotting predictions
    plot_predictions(predictions_np, targets_np, 'plot_prediction_lstm.png')
