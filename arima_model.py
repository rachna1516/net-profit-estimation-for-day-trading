import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from data_utils import (
    chronological_split,
    differencing_order,
    direction_accuracy,
    format_metrics,
    load_data,
    regression_metrics,
)

TARGET_COLUMN = "Open"


def save_acf_pacf(series: pd.Series, acf_path: str, pacf_path: str, lags: int = 40) -> None:
    """Save ACF/PACF plots used to choose the AR (p) and MA (q) orders."""
    fig = plot_acf(series, lags=lags)
    fig.savefig(acf_path)
    plt.close(fig)
    fig = plot_pacf(series, lags=lags)
    fig.savefig(pacf_path)
    plt.close(fig)


def fit_and_forecast(model, steps: int) -> np.ndarray:
    """Fit a statsmodels state-space model and forecast ``steps`` values ahead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter from the optimizer
        fitted = model.fit()
    return np.asarray(fitted.forecast(steps=steps))


def evaluate_forecast(train: pd.Series, test: pd.Series, forecast: np.ndarray) -> dict:
    """Score a price-level forecast against the test period."""
    actual = test.to_numpy(dtype=float)
    previous = np.concatenate([[train.iloc[-1]], actual[:-1]])
    metrics = regression_metrics(actual, forecast)
    metrics["direction_accuracy"] = direction_accuracy(actual, forecast, previous)
    return metrics


def plot_forecast(test: pd.Series, forecast: np.ndarray, title: str, savefile: str) -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.to_numpy(), label="Actual", color="blue")
    plt.plot(test.index, forecast, label="Predicted", color="orange")
    plt.title(title)
    plt.xlabel("Datetime")
    plt.ylabel(f"{TARGET_COLUMN} Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    Path(savefile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefile)
    plt.close(fig)


def run_arima(series: pd.Series, p: int = 3, q: int = 2, d=None, train_frac: float = 0.8):
    """Fit ARIMA(p, d, q) on the training part of ``series`` and forecast the test part.

    If ``d`` is None it is chosen with the ADF test. The model is fit on price
    *levels*; ARIMA's ``d`` handles the differencing internally, so the
    forecast is directly comparable to actual prices.
    """
    train, test = chronological_split(series, train_frac)
    if d is None:
        d = differencing_order(train)
    model = ARIMA(train.to_numpy(dtype=float), order=(p, d, q))
    forecast = fit_and_forecast(model, steps=len(test))
    return train, test, forecast, d


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARIMA forecast of the Open price.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("-p", type=int, default=3, help="AR order")
    parser.add_argument("-q", type=int, default=2, help="MA order")
    parser.add_argument("-d", type=int, default=None, help="differencing order (default: ADF test)")
    parser.add_argument("--plot", default="plot_prediction_arima.png")
    parser.add_argument("--no-diagnostics", action="store_true", help="skip ACF/PACF plots")
    args = parser.parse_args(argv)

    # To refresh the data: yfinance.download("WMT", interval="60m").to_csv("data.csv")
    series = load_data(args.data)[TARGET_COLUMN]
    train, test, forecast, d = run_arima(series, p=args.p, q=args.q, d=args.d)

    if not args.no_diagnostics:
        stationary = train.diff(d).dropna() if d else train
        save_acf_pacf(stationary, "acf.png", "pacf.png")

    metrics = evaluate_forecast(train, test, forecast)
    print(f"ARIMA({args.p}, {d}, {args.q}) - {len(test)}-step forecast")
    print(format_metrics(metrics))

    plot_forecast(test, forecast, "Actual vs Predicted (ARIMA)", args.plot)
    return metrics


if __name__ == "__main__":
    main()
