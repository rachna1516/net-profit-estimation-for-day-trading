import argparse

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arima_model import TARGET_COLUMN, evaluate_forecast, fit_and_forecast, plot_forecast, save_acf_pacf
from data_utils import chronological_split, differencing_order, format_metrics, load_data


def bars_per_day(index: pd.DatetimeIndex) -> int:
    """Most common number of bars per calendar day - the natural intraday season.

    For hourly US-market data (09:30-15:30) this is 7, not 24.
    """
    counts = pd.Series(1, index=index).groupby(index.normalize()).sum()
    if counts.empty:
        raise ValueError("Cannot infer seasonality from an empty index")
    return int(counts.mode().iloc[0])


def run_sarima(
    series: pd.Series,
    order=(3, None, 2),
    seasonal_order=(3, 1, 2, None),
    train_frac: float = 0.8,
):
    """Fit SARIMA on the training part of ``series`` and forecast the test part.

    ``None`` for ``d`` is chosen with the ADF test; ``None`` for the season
    length ``s`` is inferred as the number of bars per trading day.
    """
    train, test = chronological_split(series, train_frac)
    p, d, q = order
    P, D, Q, s = seasonal_order
    if d is None:
        d = differencing_order(train)
    if s is None:
        s = bars_per_day(series.index)

    model = SARIMAX(
        train.to_numpy(dtype=float),
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    forecast = fit_and_forecast(model, steps=len(test))
    return train, test, forecast, (p, d, q), (P, D, Q, s)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Seasonal ARIMA forecast of the Open price.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--order", type=int, nargs=3, default=[3, -1, 2], metavar=("p", "d", "q"),
                        help="non-seasonal order; d=-1 picks d with the ADF test")
    parser.add_argument("--seasonal-order", type=int, nargs=4, default=[3, 1, 2, -1],
                        metavar=("P", "D", "Q", "s"),
                        help="seasonal order; s=-1 infers bars per trading day")
    parser.add_argument("--plot", default="plot_prediction_sarima.png")
    parser.add_argument("--no-diagnostics", action="store_true", help="skip ACF/PACF/series plots")
    args = parser.parse_args(argv)

    order = tuple(None if v == -1 else v for v in args.order)
    seasonal = tuple(None if v == -1 and i == 3 else v for i, v in enumerate(args.seasonal_order))

    # To refresh the data: yfinance.download("WMT", interval="60m").to_csv("data.csv")
    series = load_data(args.data)[TARGET_COLUMN]
    train, test, forecast, order, seasonal = run_sarima(series, order, seasonal)

    if not args.no_diagnostics:
        stationary = train.diff(order[1]).dropna() if order[1] else train
        fig = plt.figure(figsize=(12, 6))
        plt.plot(stationary, label=f"{TARGET_COLUMN} (differenced x{order[1]})")
        plt.title("Training series used to inspect seasonality")
        plt.xlabel("Datetime")
        plt.legend()
        fig.savefig("seasonality.png")
        plt.close(fig)
        save_acf_pacf(stationary, "sarima_acf.png", "sarima_pacf.png")

    metrics = evaluate_forecast(train, test, forecast)
    print(f"SARIMA{order}x{seasonal} - {len(test)}-step forecast")
    print(format_metrics(metrics))

    plot_forecast(test, forecast, "Actual vs Predicted (SARIMA)", args.plot)
    return metrics


if __name__ == "__main__":
    main()
