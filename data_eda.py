import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

print(df.describe())
print(f"number of null values: {df.isna().sum().sum()}")

# normalizing data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
df_scaled = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], index=df.index)

# Plot time series
plt.figure(figsize=(10, 6))
plt.plot(df_scaled['Open'])
plt.title('Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Scaled Open Price')
plt.savefig('EDA_distribution_graphs/open_trend.png')

plt.figure(figsize=(10, 6))
plt.plot(df_scaled['Close'])
plt.title('closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Scaled close Price')
plt.savefig('EDA_distribution_graphs/close_trend.png')

plt.figure(figsize=(10, 6))
plt.plot(df_scaled['High'])
plt.title('High Price Over Time')
plt.xlabel('Date')
plt.ylabel('Scaled High Price')
plt.savefig('EDA_distribution_graphs/high_trend.png')

plt.figure(figsize=(10, 6))
plt.plot(df_scaled['Low'])
plt.title('Low Price Over Time')
plt.xlabel('Date')
plt.ylabel('Scaled Low Price')
plt.savefig('EDA_distribution_graphs/low_trend.png')

# plotting correlation matrix
corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('EDA_distribution_graphs/correlation_matrix.png')

# plotting volatality (SD of returns)
df['Returns'] = df['Open'].pct_change()
df['Returns'].rolling(window=21).std().plot(figsize=(14, 7))
plt.title('Rolling 21-days Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.savefig('EDA_distribution_graphs/open_volatality.png')

# Moving Averages
df['SMA_20'] = df['Open'].rolling(window=20).mean()
df['SMA_50'] = df['Open'].rolling(window=50).mean()
# Daily Returns
df['Returns'] = df['Open'].pct_change()
# Cumulative Returns
df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
# Plot historical prices with moving averages
plt.figure(figsize=(14, 7))
plt.plot(df['Open'], label='Open Price')
plt.plot(df['SMA_20'], label='20-Day SMA')
plt.plot(df['SMA_50'], label='50-Day SMA')
plt.title('Open Price with Moving Averages')
plt.legend()
plt.savefig('EDA_distribution_graphs/open_price_moving_avgs.png')
# Returns distribution
sns.histplot(df['Returns'].dropna(), bins=50, kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.savefig('EDA_distribution_graphs/returns_dist.png')