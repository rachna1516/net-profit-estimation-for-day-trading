import numpy as np
import pandas as pd
import pytest

from data_utils import DATETIME_COLUMN, FEATURE_COLUMNS

BAR_TIMES = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]


def make_ohlcv(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly OHLCV data with 7 bars per business day (like data.csv)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{day.date()} {t}") for day in days for t in BAR_TIMES], name=DATETIME_COLUMN
    )
    n = len(index)
    close = 50 + np.cumsum(rng.normal(0, 0.3, n)) + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 7)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 0.2, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.2, n)
    volume = rng.integers(100_000, 1_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume},
        index=index,
    )[FEATURE_COLUMNS]


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def ohlcv_csv(tmp_path, ohlcv_df):
    path = tmp_path / "data.csv"
    ohlcv_df.reset_index().to_csv(path, index=False)
    return path
