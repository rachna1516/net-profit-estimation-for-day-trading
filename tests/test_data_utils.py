import numpy as np
import pandas as pd
import pytest

from data_utils import (
    FEATURE_COLUMNS,
    chronological_split,
    create_sequences,
    differencing_order,
    direction_accuracy,
    format_metrics,
    load_data,
    regression_metrics,
)


# ---------- load_data ----------

def test_load_data_parses_index_and_columns(ohlcv_csv):
    df = load_data(ohlcv_csv)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == FEATURE_COLUMNS
    assert df.index.is_monotonic_increasing


def test_load_data_sorts_dedupes_and_drops_nans(tmp_path, ohlcv_df):
    raw = ohlcv_df.reset_index()
    shuffled = pd.concat([raw.iloc[::-1], raw.iloc[[0]]])  # reversed + one duplicate timestamp
    shuffled.loc[shuffled.index[5], "Close"] = np.nan
    path = tmp_path / "messy.csv"
    shuffled.to_csv(path, index=False)

    df = load_data(path)
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert not df.isna().any().any()
    assert len(df) == len(ohlcv_df) - 1  # only the NaN row is lost


def test_load_data_rejects_missing_columns(tmp_path, ohlcv_df):
    path = tmp_path / "bad.csv"
    ohlcv_df.drop(columns=["Volume"]).reset_index().to_csv(path, index=False)
    with pytest.raises(ValueError, match="Volume"):
        load_data(path)


def test_load_real_data_file():
    """Sanity check on the committed dataset."""
    df = load_data("data.csv")
    assert len(df) > 1000
    assert (df["High"] >= df["Low"]).all()
    assert (df["Volume"] >= 0).all()


# ---------- chronological_split ----------

def test_chronological_split_keeps_order():
    data = np.arange(10)
    train, test = chronological_split(data, 0.8)
    np.testing.assert_array_equal(train, np.arange(8))
    np.testing.assert_array_equal(test, [8, 9])


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_chronological_split_rejects_bad_fraction(frac):
    with pytest.raises(ValueError):
        chronological_split(np.arange(10), frac)


def test_chronological_split_rejects_tiny_data():
    with pytest.raises(ValueError):
        chronological_split(np.arange(1), 0.5)


# ---------- create_sequences ----------

def test_create_sequences_shapes_and_alignment():
    data = np.arange(20, dtype=float).reshape(10, 2)  # col0 = 0,2,4,...
    X, y = create_sequences(data, seq_length=3, target_col=0)
    assert X.shape == (7, 3, 2)
    assert y.shape == (7,)
    np.testing.assert_array_equal(X[0], data[0:3])
    # target is the row right after the window - no overlap with the input
    assert y[0] == data[3, 0]
    assert y[-1] == data[9, 0]


def test_create_sequences_other_target_column():
    data = np.arange(20, dtype=float).reshape(10, 2)
    _, y = create_sequences(data, seq_length=2, target_col=1)
    np.testing.assert_array_equal(y, data[2:, 1])


def test_create_sequences_too_short_returns_empty():
    X, y = create_sequences(np.zeros((3, 4)), seq_length=5)
    assert X.shape == (0, 5, 4)
    assert y.shape == (0,)


@pytest.mark.parametrize("kwargs", [{"seq_length": 0}, {"seq_length": 2, "target_col": 9}])
def test_create_sequences_validates_args(kwargs):
    with pytest.raises(ValueError):
        create_sequences(np.zeros((10, 3)), **kwargs)


def test_create_sequences_rejects_1d():
    with pytest.raises(ValueError):
        create_sequences(np.zeros(10), 2)


# ---------- differencing_order ----------

def test_differencing_order_stationary_series_is_zero():
    rng = np.random.default_rng(0)
    assert differencing_order(pd.Series(rng.normal(size=500))) == 0


def test_differencing_order_random_walk_is_one():
    rng = np.random.default_rng(0)
    assert differencing_order(pd.Series(np.cumsum(rng.normal(size=500)))) == 1


def test_differencing_order_too_short():
    with pytest.raises(ValueError):
        differencing_order(pd.Series([1.0, 2.0, 3.0]))


# ---------- metrics ----------

def test_regression_metrics_perfect_prediction():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_regression_metrics_known_values():
    m = regression_metrics([0, 0, 0, 0], [1, -1, 1, -1])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)


def test_regression_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        regression_metrics([1, 2, 3], [1, 2])


def test_regression_metrics_empty():
    with pytest.raises(ValueError):
        regression_metrics([], [])


def test_direction_accuracy_counts_matching_moves():
    previous = [10, 10, 10, 10]
    actual = [11, 9, 11, 9]      # up, down, up, down
    predicted = [12, 8, 9, 11]   # up, down, down, up
    assert direction_accuracy(actual, predicted, previous) == pytest.approx(0.5)


def test_direction_accuracy_ignores_flat_steps():
    assert direction_accuracy([10, 11], [99, 12], [10, 10]) == 1.0


def test_direction_accuracy_all_flat_is_nan():
    assert np.isnan(direction_accuracy([1, 1], [2, 0], [1, 1]))


def test_format_metrics():
    text = format_metrics({"mae": 1.23456, "rmse": 2.0})
    assert "Mean Absolute Error (MAE): 1.2346" in text
    assert "Root Mean Squared Error (RMSE): 2.0000" in text
