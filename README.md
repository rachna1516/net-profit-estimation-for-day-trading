Stock-day-trading---Time-series-Analysis-with-Machine-learning

The objective of this project is to implement time series analysis for day trading using Autoregressive Integrated Moving Average (ARIMA) / Seasonal ARIMA (SARIMA) models and Long Short-Term Memory (LSTM) neural networks.

## Introduction

A time series analysis technique is used to predict short-term price movements in financial markets. The objective is to combine statistical analysis (ARIMA/SARIMA) with machine learning (LSTM) to identify both linear and non-linear patterns in the data.

## Features

- Feature engineering and data preprocessing for financial time series
- LSTM networks for sequence prediction
- Linear time series forecasting with ARIMA and SARIMA
- Time series and model performance visualization
- Interactive Streamlit UI

## Setup

Python 3.9+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

(`environment.yml` is the original Linux/CUDA conda environment.)

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Tabs: **Data & EDA** (charts and statistics), **ARIMA**, **SARIMA**, **LSTM** (set parameters and run), and **Compare** (metrics and forecasts from every model you ran, side by side). Upload your own CSV from the sidebar, or use the bundled `data.csv`.

### Command line

```bash
python data_eda.py                     # charts into EDA_distribution_graphs/
python arima_model.py -p 3 -q 2        # d is chosen with the ADF test unless -d is given
python sarima_model.py                 # season length defaults to bars per trading day (7 for hourly data)
python LSTM_implementation.py --epochs 50 --seq-length 10
```

Each script accepts `--help` for all options.

### Tests

```bash
pytest
```

## Models

- LSTM (Long Short-Term Memory): predicts the next bar's open price from a sliding window of all features.
- ARIMA (Autoregressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)

ARIMA/SARIMA forecast the whole test period from the end of the training data. The LSTM predicts one step ahead using actual recent bars, so its scores are not directly comparable to the ARIMA/SARIMA scores.

## Data

Input CSV columns: `Datetime, Open, High, Low, Close, Adj Close, Volume`.
The bundled `data.csv` holds hourly WMT bars (2022-09 to 2024-09). To refresh it:

```python
import yfinance as yf
yf.download("WMT", interval="60m").to_csv("data.csv")
```

## Results

Models are scored on a chronological 80/20 train/test split using Mean Absolute Error (MAE), Root Mean Square Error (RMSE), R², and Direction Accuracy (share of bars where the predicted up/down move matches the actual one). Each script saves a chart of predicted vs actual values.
