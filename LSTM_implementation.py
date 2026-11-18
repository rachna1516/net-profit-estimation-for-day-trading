import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    FEATURE_COLUMNS,
    create_sequences,
    direction_accuracy,
    format_metrics,
    load_data,
    regression_metrics,
)
from dataloader import StockDataset
from model import Gatedmodel

TARGET_COLUMN = "Open"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(df, seq_length: int, train_frac: float = 0.8):
    """Scale features and build train/test sequences without look-ahead leakage.

    The scaler is fit on the training rows only. A sample belongs to the test
    set when its *target* row falls in the test period; its input window may
    reach back into the training period, which is information available at
    prediction time.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")
    split_row = int(len(df) * train_frac)
    if split_row <= seq_length or split_row >= len(df):
        raise ValueError("Not enough rows for the requested seq_length/train_frac")

    target_col = FEATURE_COLUMNS.index(TARGET_COLUMN)
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURE_COLUMNS].iloc[:split_row])
    scaled = scaler.transform(df[FEATURE_COLUMNS])

    X, y = create_sequences(scaled, seq_length, target_col=target_col)
    # Sample i predicts row i + seq_length.
    split_sample = split_row - seq_length
    return {
        "X_train": X[:split_sample],
        "y_train": y[:split_sample],
        "X_test": X[split_sample:],
        "y_test": y[split_sample:],
        # last known (scaled) target value before each test target, for direction accuracy
        "prev_test": scaled[split_row - 1 : len(df) - 1, target_col],
        "scaler": scaler,
        "target_col": target_col,
    }


def inverse_target(values, scaler: MinMaxScaler, target_col: int) -> np.ndarray:
    """Undo MinMax scaling for a single column."""
    values = np.asarray(values, dtype=float).ravel()
    return values * scaler.data_range_[target_col] + scaler.data_min_[target_col]


def train_model(model, dataloader, criterion, optimizer, device="cpu") -> float:
    """Run one training epoch; returns the mean batch loss."""
    model.train()
    total_loss, n_batches = 0.0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_model(model, X_test, y_test, device="cpu"):
    """Predict on the test set; returns (predictions, targets) as 1-D arrays."""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32, device=device)
        predictions = model(inputs).cpu().numpy().ravel()
    return predictions, np.asarray(y_test, dtype=float).ravel()


def plot_predictions(predictions, targets, savefile, ylabel="Open Price"):
    fig = plt.figure(figsize=(14, 7))
    plt.plot(targets, label="Actual", color="blue")
    plt.plot(predictions, label="Predicted", color="red")
    plt.title("Stock Price Prediction vs Actual (LSTM)")
    plt.xlabel("Time step (test set)")
    plt.ylabel(ylabel)
    plt.legend()
    Path(savefile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefile)
    plt.close(fig)


def train_and_forecast(df, seq_length=10, epochs=50, batch_size=32, lr=0.001, seed=42,
                       train_frac=0.8, progress=None):
    """Train the LSTM and predict the test period in price units.

    ``progress`` is an optional callback ``progress(epoch, epochs, loss)``.
    Returns a dict with ``predictions``, ``targets``, ``index`` (test timestamps),
    ``losses`` and ``metrics``.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_data(df, seq_length, train_frac)

    loader = DataLoader(StockDataset(data["X_train"], data["y_train"]), batch_size=batch_size, shuffle=True)
    model = Gatedmodel(data["X_train"].shape[2], 64, 32, 1).to(device)
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        losses.append(train_model(model, loader, criterion, optimizer, device))
        if progress is not None:
            progress(epoch + 1, epochs, losses[-1])

    pred_scaled, target_scaled = evaluate_model(model, data["X_test"], data["y_test"], device)
    scaler, col = data["scaler"], data["target_col"]
    predictions = inverse_target(pred_scaled, scaler, col)
    targets = inverse_target(target_scaled, scaler, col)
    previous = inverse_target(data["prev_test"], scaler, col)

    metrics = regression_metrics(targets, predictions)
    metrics["direction_accuracy"] = direction_accuracy(targets, predictions, previous)
    return {
        "predictions": predictions,
        "targets": targets,
        "index": df.index[len(df) - len(targets):],
        "losses": losses,
        "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train an LSTM to forecast the next Open price.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--seq-length", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", default="plot_prediction_lstm.png")
    args = parser.parse_args(argv)

    # To refresh the data: yfinance.download("WMT", interval="60m").to_csv("data.csv")
    df = load_data(args.data)
    with tqdm(total=args.epochs, desc="Training") as bar:
        result = train_and_forecast(
            df, args.seq_length, args.epochs, args.batch_size, args.lr, args.seed,
            progress=lambda *_: bar.update(1),
        )

    print(format_metrics(result["metrics"]))
    plot_predictions(result["predictions"], result["targets"], args.plot)
    return result["metrics"]


if __name__ == "__main__":
    main()
