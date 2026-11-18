import numpy as np
import pandas as pd
import pytest

import arima_model
import sarima_model


def test_run_arima_forecast_length_matches_test(ohlcv_df):
    train, test, forecast, d = arima_model.run_arima(ohlcv_df["Open"], p=1, q=1)
    assert len(forecast) == len(test)
    assert len(train) + len(test) == len(ohlcv_df)
    assert d in (0, 1, 2)
    assert np.isfinite(forecast).all()


def test_run_arima_forecasts_price_levels_not_differences(ohlcv_df):
    """Regression test: the old code differenced twice and forecast changes (~0)."""
    _, test, forecast, _ = arima_model.run_arima(ohlcv_df["Open"], p=1, q=1)
    assert abs(forecast.mean() - test.mean()) < 0.5 * test.mean()


def test_run_arima_handles_already_stationary_series():
    """Regression test: the old code raised NameError when the ADF test passed."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2023-01-01", periods=300, freq="h")
    series = pd.Series(50 + rng.normal(size=300), index=index)
    _, test, forecast, d = arima_model.run_arima(series, p=1, q=0)
    assert d == 0
    assert len(forecast) == len(test)


def test_run_arima_respects_explicit_d(ohlcv_df):
    *_, d = arima_model.run_arima(ohlcv_df["Open"], p=1, q=0, d=1)
    assert d == 1


def test_evaluate_forecast_perfect():
    index = pd.date_range("2023-01-01", periods=5, freq="h")
    train = pd.Series([1.0, 2.0], index=index[:2])
    test = pd.Series([3.0, 2.0, 4.0], index=index[2:])
    m = arima_model.evaluate_forecast(train, test, test.to_numpy())
    assert m["mae"] == 0 and m["rmse"] == 0
    assert m["direction_accuracy"] == 1.0


def test_bars_per_day_detects_trading_hours(ohlcv_df):
    assert sarima_model.bars_per_day(ohlcv_df.index) == 7


def test_bars_per_day_on_real_data():
    from data_utils import load_data

    assert sarima_model.bars_per_day(load_data("data.csv").index) == 7


def test_run_sarima_infers_season_and_forecasts(ohlcv_df):
    train, test, forecast, order, seasonal = sarima_model.run_sarima(
        ohlcv_df["Open"], order=(1, None, 0), seasonal_order=(1, 0, 0, None)
    )
    assert seasonal[3] == 7
    assert order[1] is not None
    assert len(forecast) == len(test)
    assert np.isfinite(forecast).all()


@pytest.mark.parametrize("module", [arima_model, sarima_model])
def test_main_end_to_end(module, ohlcv_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # diagnostic plots are written to the cwd
    plot = tmp_path / "forecast.png"
    args = ["--data", str(ohlcv_csv), "--plot", str(plot)]
    if module is sarima_model:
        args += ["--order", "1", "-1", "0", "--seasonal-order", "1", "0", "0", "-1"]
    else:
        args += ["-p", "1", "-q", "1"]
    metrics = module.main(args)
    assert plot.exists()
    assert set(metrics) == {"mae", "rmse", "r2", "direction_accuracy"}
