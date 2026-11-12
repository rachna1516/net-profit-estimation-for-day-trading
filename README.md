Stock-day-trading---Time-series-Analysis-with-Machine-learning

The objective of this project is to implement time series analysis for day trading using Autoregressive Integrated Moving Average (ARIMA) / Seasonal ARIMA (SARIMA) models and Long Short-Term Memory (LSTM) neural networks.

Introduction 

A time series analysis technique will be used to predict short-term price movements in financial markets Our objective is to integrate statistical analysis (ARIMA/SARIMA) with machine learning (LSTM) in order to identify both linear and non-linear patterns in the data.


Features:

Feature engineering and data preprocessing for financial time series
The implementation of LSTM networks for sequence prediction
Forecasting linear time series with ARIMA and SARIMA  
Time series and model performance visualization tools

Requirements

Python 3.9+
Pandas
NumPy
Matplotlib
Statsmodels
Scikit-learn

Models
LSTM (Long Short-Term Memory)
ARIMA (Autoregressive Integrated Moving Average)
SARIMA (Seasonal ARIMA)

Data
The project expects input data in CSV format with the following columns:

Date
Open
High
Low
Close
Volume 

Results
Metrics like Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Direction Accuracy are used to assess the performance of the model. The results directory produces visualizations comparing the outcomes that were predicted to the actual values.