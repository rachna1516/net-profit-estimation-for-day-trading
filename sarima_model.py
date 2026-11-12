import pandas as pd
import numpy as np
import yfinance
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == "__main__":
    # downloading the latest data
    # df = yf.download("WMT", interval="60m")  # interval 60 minutes
    # df.to_csv('data.csv')

    # loading the data
    df = pd.read_csv('data.csv')
    
    # preprocessing features and normalization
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    df_train = df[:int(df.shape[0] * 0.8)]
    df_test   = df[int(df.shape[0] * 0.8):]

    opening_price = df_train['Open']
    result = adfuller(opening_price.dropna())
    if result[1] > 0.05:
        diff_opening_prices = opening_price.diff().dropna()

    # Plot the opening prices
    plt.figure(figsize=(12, 6))
    plt.plot(diff_opening_prices, label='Opening Price')
    plt.title('Opening Price Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Opening Price')
    plt.legend()
    plt.savefig("seasonality.png")

    plot_acf(diff_opening_prices)
    plt.savefig('sarima_acf.png')
    plot_pacf(diff_opening_prices)
    plt.savefig('sarima_pacf.png')

    # SARIMA parameters
    Q = q = 2
    P = p = 3
    D = d = 1   # since done differencing once to make series stationary
    s = 24

    model = SARIMAX(diff_opening_prices,
                 order=(p, d, q),
                 seasonal_order=(P, D, Q, s),
                 enforce_stationarity=False,
                 enforce_invertibility=False)
    model_fit = model.fit()

    # Forecast the number of test steps
    forecast = model_fit.get_forecast(steps=698)
    forecast_index = pd.date_range(start=diff_opening_prices.index[-1] + pd.Timedelta(hours=1), periods=698, freq='H')
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # testing
    gt = df_test['Open'].diff().dropna()

    df_ground_truth = gt.reset_index(drop=True)
    df_predicted = forecast_values.reset_index(drop=True)

    mae = mean_absolute_error(df_ground_truth, df_predicted)
    rmse = np.sqrt(mean_squared_error(df_ground_truth, df_predicted))
    r2 = r2_score(df_ground_truth, df_predicted)

    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score: {r2}')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df_ground_truth, label='Actual values of test', marker='o', color='blue')
    plt.plot(df_predicted, label='Predicted values of test', marker='o', color='orange')

    plt.title('Actual Values vs Predicted Values')
    plt.xlabel('Datetime')
    plt.ylabel('Open Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig("plot_prediction_sarima.png")