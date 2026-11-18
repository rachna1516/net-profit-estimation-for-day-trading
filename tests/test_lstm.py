import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import LSTM_implementation as lstm
from dataloader import StockDataset, lstm_create_sequences
from model import Gatedmodel


# ---------- model ----------

def test_model_output_shape():
    model = Gatedmodel(input_size=6, hidden_size1=16, hidden_size2=8, output_size=1)
    out = model(torch.randn(4, 10, 6))
    assert out.shape == (4, 1)


def test_model_builds_without_dropout_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Gatedmodel(6, 16, 8, 1)


def test_model_eval_is_deterministic():
    model = Gatedmodel(6, 16, 8, 1).eval()
    x = torch.randn(2, 5, 6)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


# ---------- dataset ----------

def test_dataset_tensors_and_shapes():
    X, y = lstm_create_sequences(np.random.rand(20, 6), 5)
    ds = StockDataset(X, y)
    assert len(ds) == 15
    features, target = ds[0]
    assert features.shape == (5, 6) and features.dtype == torch.float32
    assert target.shape == (1,)


def test_dataset_rejects_length_mismatch():
    with pytest.raises(ValueError):
        StockDataset(np.zeros((5, 3, 2)), np.zeros(4))


def test_dataloader_batches():
    X, y = lstm_create_sequences(np.random.rand(50, 6), 5)
    features, targets = next(iter(DataLoader(StockDataset(X, y), batch_size=8)))
    assert features.shape == (8, 5, 6)
    assert targets.shape == (8, 1)


# ---------- data preparation ----------

def test_prepare_data_split_sizes(ohlcv_df):
    seq = 10
    data = lstm.prepare_data(ohlcv_df, seq, train_frac=0.8)
    split_row = int(len(ohlcv_df) * 0.8)
    assert len(data["X_train"]) == split_row - seq
    assert len(data["X_train"]) + len(data["X_test"]) == len(ohlcv_df) - seq
    assert len(data["y_test"]) == len(ohlcv_df) - split_row
    assert len(data["prev_test"]) == len(data["y_test"])


def test_prepare_data_has_no_test_leakage_in_scaler(ohlcv_df):
    """Scaler must be fit on training rows only."""
    data = lstm.prepare_data(ohlcv_df, 10, train_frac=0.8)
    split_row = int(len(ohlcv_df) * 0.8)
    train_open = ohlcv_df["Open"].iloc[:split_row]
    assert data["scaler"].data_min_[data["target_col"]] == pytest.approx(train_open.min())
    assert data["scaler"].data_max_[data["target_col"]] == pytest.approx(train_open.max())
    # training targets therefore lie within [0, 1]
    assert data["y_train"].min() >= 0 and data["y_train"].max() <= 1


def test_prepare_data_test_targets_match_real_prices(ohlcv_df):
    data = lstm.prepare_data(ohlcv_df, 10)
    split_row = int(len(ohlcv_df) * 0.8)
    restored = lstm.inverse_target(data["y_test"], data["scaler"], data["target_col"])
    np.testing.assert_allclose(restored, ohlcv_df["Open"].iloc[split_row:].to_numpy())
    prev = lstm.inverse_target(data["prev_test"], data["scaler"], data["target_col"])
    np.testing.assert_allclose(prev, ohlcv_df["Open"].iloc[split_row - 1 : -1].to_numpy())


def test_prepare_data_rejects_seq_longer_than_train(ohlcv_df):
    with pytest.raises(ValueError):
        lstm.prepare_data(ohlcv_df.iloc[:20], seq_length=50)


# ---------- training / evaluation ----------

def test_training_reduces_loss(ohlcv_df):
    lstm.set_seed(0)
    data = lstm.prepare_data(ohlcv_df, 10)
    loader = DataLoader(StockDataset(data["X_train"], data["y_train"]), batch_size=32, shuffle=True)
    model = Gatedmodel(6, 16, 8, 1, dropout=0.0)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    first = lstm.train_model(model, loader, criterion, optimizer)
    for _ in range(15):
        last = lstm.train_model(model, loader, criterion, optimizer)
    assert last < first


def test_evaluate_model_returns_flat_arrays(ohlcv_df):
    data = lstm.prepare_data(ohlcv_df, 10)
    model = Gatedmodel(6, 8, 4, 1)
    preds, targets = lstm.evaluate_model(model, data["X_test"], data["y_test"])
    assert preds.shape == targets.shape == (len(data["y_test"]),)


def test_set_seed_reproducible():
    lstm.set_seed(123)
    a = Gatedmodel(6, 8, 4, 1)(torch.ones(1, 3, 6))
    lstm.set_seed(123)
    b = Gatedmodel(6, 8, 4, 1)(torch.ones(1, 3, 6))
    assert torch.equal(a, b)


def test_main_end_to_end(ohlcv_csv, tmp_path):
    plot = tmp_path / "lstm.png"
    metrics = lstm.main(["--data", str(ohlcv_csv), "--epochs", "2", "--plot", str(plot)])
    assert plot.exists()
    assert set(metrics) == {"mae", "rmse", "r2", "direction_accuracy"}
    assert metrics["mae"] >= 0 and np.isfinite(metrics["rmse"])
