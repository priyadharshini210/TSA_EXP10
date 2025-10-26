# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 26/10/2025
## Name: Priyadharshini P
## Reg no: 212223240128

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# === Load the dataset ===
data = pd.read_csv("silver.csv")

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop missing or invalid dates
data.dropna(subset=['Date'], inplace=True)

# Sort by date
data = data.sort_values(by='Date')

# Set 'Date' as index
data.set_index('Date', inplace=True)

# Display first few rows
print(data.head())

# === Select the target variable for time series analysis ===
target_col = 'USD'

# === Plot the time series ===
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[target_col], label=target_col, color='blue')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.title(f'{target_col} Time Series')
plt.legend()
plt.grid()
plt.show()

# === Function to check stationarity ===
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# === Check stationarity of the series ===
print("\n--- Stationarity Test for Value ---")
check_stationarity(data[target_col].dropna())

# === Plot ACF and PACF ===
plt.figure(figsize=(10, 4))
plot_acf(data[target_col].dropna(), lags=30)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(data[target_col].dropna(), lags=30)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# === Train-Test Split ===
train_size = int(len(data) * 0.8)
train, test = data[target_col][:train_size], data[target_col][train_size:]

# === Build and fit SARIMA model ===
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# === Forecast ===
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# === Evaluate performance ===
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.4f}')

# === Plot predictions vs actuals ===
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel(target_col)
plt.title(f'SARIMA Model Predictions for {target_col}')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:

<img width="1076" height="640" alt="image" src="https://github.com/user-attachments/assets/ae42882a-9642-4145-9976-bf97da52d622" />
<img width="657" height="571" alt="image" src="https://github.com/user-attachments/assets/bf53d3d6-481f-43cf-8434-f9197801d506" />
<img width="572" height="443" alt="image" src="https://github.com/user-attachments/assets/47dee3f6-7dac-41cb-974c-245d496e7ac2" />
<img width="992" height="532" alt="image" src="https://github.com/user-attachments/assets/b812cab9-097f-43d4-978e-3c887d46a2c4" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
