import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from data_utils import FEATURE_COLUMNS, load_data


def add_indicators(df: pd.DataFrame, short_window: int = 20, long_window: int = 50,
                   vol_window: int = 21) -> pd.DataFrame:
    """Return a copy of ``df`` with returns, moving averages and volatility columns."""
    out = df.copy()
    out["Returns"] = out["Open"].pct_change()
    out["Cumulative_Returns"] = (1 + out["Returns"]).cumprod()
    out[f"SMA_{short_window}"] = out["Open"].rolling(window=short_window).mean()
    out[f"SMA_{long_window}"] = out["Open"].rolling(window=long_window).mean()
    out["Volatility"] = out["Returns"].rolling(window=vol_window).std()
    return out


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def run_eda(df: pd.DataFrame, out_dir="EDA_distribution_graphs") -> list:
    """Print summary statistics and save EDA charts; returns the saved file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    print(df.describe())
    print(f"number of null values: {df.isna().sum().sum()}")

    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(df[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS, index=df.index
    )

    for column, filename in [("Open", "open_trend.png"), ("Close", "close_trend.png"),
                             ("High", "high_trend.png"), ("Low", "low_trend.png")]:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(scaled[column])
        plt.title(f"{column} Price Over Time")
        plt.xlabel("Date")
        plt.ylabel(f"Scaled {column} Price")
        saved.append(_save(fig, out_dir, filename))

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(df[["Open", "High", "Low", "Close", "Volume"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    saved.append(_save(fig, out_dir, "correlation_matrix.png"))

    ind = add_indicators(df)

    fig = plt.figure(figsize=(14, 7))
    plt.plot(ind["Volatility"])
    plt.title("Rolling 21-bar Volatility of Open Returns")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    saved.append(_save(fig, out_dir, "open_volatality.png"))

    fig = plt.figure(figsize=(14, 7))
    plt.plot(ind["Open"], label="Open Price")
    plt.plot(ind["SMA_20"], label="20-bar SMA")
    plt.plot(ind["SMA_50"], label="50-bar SMA")
    plt.title("Open Price with Moving Averages")
    plt.legend()
    saved.append(_save(fig, out_dir, "open_price_moving_avgs.png"))

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(ind["Returns"].dropna(), bins=50, kde=True)
    plt.title("Distribution of Hourly Returns")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    saved.append(_save(fig, out_dir, "returns_dist.png"))

    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exploratory data analysis charts.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--out-dir", default="EDA_distribution_graphs")
    args = parser.parse_args(argv)
    run_eda(load_data(args.data), args.out_dir)


if __name__ == "__main__":
    main()
