"""Shared data loading, preprocessing and evaluation helpers.

All model scripts (ARIMA, SARIMA, LSTM) and the EDA script use these helpers so
that loading, splitting and scoring behave the same way everywhere.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

DATETIME_COLUMN = "Datetime"
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]

PathLike = Union[str, Path]


def load_data(path: PathLike = "data.csv") -> pd.DataFrame:
    """Load an OHLCV CSV into a clean, chronologically sorted DataFrame.

    - Validates that the datetime column and all feature columns exist.
    - Parses datetimes and uses them as the index.
    - Sorts by time, drops duplicate timestamps (keeping the last row).
    - Drops rows with missing values.
    """
    df = pd.read_csv(path)

    required = [DATETIME_COLUMN, *FEATURE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
    df = df.set_index(DATETIME_COLUMN).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[FEATURE_COLUMNS].dropna()

    if df.empty:
        raise ValueError("Input data contains no usable rows after cleaning")
    return df


def chronological_split(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], train_frac: float = 0.8
) -> Tuple:
    """Split time-ordered data into train/test without shuffling."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")
    split_idx = int(len(data) * train_frac)
    if split_idx == 0 or split_idx == len(data):
        raise ValueError("Not enough data to create non-empty train and test sets")
    return data[:split_idx], data[split_idx:]


def create_sequences(
    data: np.ndarray, seq_length: int, target_col: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences for sequence models.

    Each sample ``X[i]`` is ``data[i : i + seq_length]`` (all features) and its
    target ``y[i]`` is ``data[i + seq_length, target_col]`` - the value of the
    target column on the step immediately after the window.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (time, features), got shape {data.shape}")
    if seq_length < 1:
        raise ValueError(f"seq_length must be >= 1, got {seq_length}")
    if not 0 <= target_col < data.shape[1]:
        raise ValueError(f"target_col {target_col} out of range for {data.shape[1]} features")

    n_samples = len(data) - seq_length
    if n_samples <= 0:
        return (
            np.empty((0, seq_length, data.shape[1]), dtype=data.dtype),
            np.empty((0,), dtype=data.dtype),
        )

    xs = np.stack([data[i : i + seq_length] for i in range(n_samples)])
    ys = data[seq_length:, target_col]
    return xs, ys


def differencing_order(series: pd.Series, alpha: float = 0.05, max_d: int = 2) -> int:
    """Return how many times ``series`` must be differenced to pass the ADF test.

    Uses the Augmented Dickey-Fuller test: a p-value <= ``alpha`` means the
    series is considered stationary. Stops at ``max_d``.
    """
    current = series.dropna()
    for d in range(max_d + 1):
        if len(current) < 10:
            raise ValueError("Series too short to test for stationarity")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # newer statsmodels return-type notice
            p_value = adfuller(current)[1]
        if p_value <= alpha:
            return d
        if d < max_d:
            current = current.diff().dropna()
    return max_d


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """MAE, RMSE and R^2 for two equal-length sequences."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }


def direction_accuracy(
    y_true: Sequence[float], y_pred: Sequence[float], previous: Sequence[float]
) -> float:
    """Fraction of steps where the predicted move direction matches the actual one.

    The move for step ``t`` is measured against ``previous[t]`` (the last known
    actual value before step ``t``). Steps with no actual movement are ignored.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    previous = np.asarray(previous, dtype=float).ravel()
    if not (y_true.shape == y_pred.shape == previous.shape):
        raise ValueError("y_true, y_pred and previous must have the same shape")

    actual_move = np.sign(y_true - previous)
    predicted_move = np.sign(y_pred - previous)
    moved = actual_move != 0
    if not moved.any():
        return float("nan")
    return float(np.mean(actual_move[moved] == predicted_move[moved]))


def format_metrics(metrics: Dict[str, float]) -> str:
    """Human-readable, one-metric-per-line summary."""
    labels = {
        "mae": "Mean Absolute Error (MAE)",
        "rmse": "Root Mean Squared Error (RMSE)",
        "r2": "R-squared (R2)",
        "direction_accuracy": "Direction Accuracy",
    }
    return "\n".join(f"{labels.get(k, k)}: {v:.4f}" for k, v in metrics.items())
