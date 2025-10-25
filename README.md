# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 25-10-2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Load CSV
data = pd.read_csv('/content/plane_crash.csv')
print("Columns in CSV:", data.columns)

# Identify numeric column for crashes
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    raise ValueError("No numeric column found for crash counts!")
crash_col = numeric_cols[0]
print("Using numeric column for crashes:", crash_col)

# Identify Date column
date_cols = data.select_dtypes(include=['object', 'datetime']).columns.tolist()
if len(date_cols) == 0:
    raise ValueError("No Date column found!")
date_col = date_cols[0]
data[date_col] = pd.to_datetime(data[date_col])
print("Using Date column:", date_col)

# Plot original data
plt.figure(figsize=(12,6))
plt.plot(data[crash_col], label='Original Crash Data')
plt.title('Original Plane Crash Data')
plt.xlabel('Date')
plt.ylabel('Number of Crashes')
plt.legend()
plt.grid()
plt.show()

# Moving Average
rolling_mean_5 = data[crash_col].rolling(window=5).mean()
rolling_mean_10 = data[crash_col].rolling(window=10).mean()

print("Rolling Mean (window=5):\n", rolling_mean_5.head(10))
print("\nRolling Mean (window=10):\n", rolling_mean_10.head(20))

plt.figure(figsize=(12,6))
plt.plot(data[crash_col], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Plane Crash Data')
plt.xlabel('Date')
plt.ylabel('Number of Crashes')
plt.legend()
plt.grid()
plt.show()

# Resample monthly
data_monthly = data.resample('MS', on=date_col).sum()

# Exponential Smoothing (additive seasonality)
x = int(len(data_monthly) * 0.8)
train_data = data_monthly[crash_col][:x]
test_data = data_monthly[crash_col][x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(12,6))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual Evaluation of Plane Crash Data')
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("RMSE of Test Predictions:", rmse)

# Forecast next period
model = ExponentialSmoothing(data_monthly[crash_col], trend='add', seasonal='add', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data_monthly)/4))

ax = data_monthly[crash_col].plot(figsize=(12,6))
predictions.plot(ax=ax)
ax.legend(["data_monthly", "predictions"])
ax.set_xlabel('Months')
ax.set_ylabel('Number of Crashes')
ax.set_title('Prediction of Plane Crash Data')
plt.show()


```

### OUTPUT:

<img width="1313" height="662" alt="image" src="https://github.com/user-attachments/assets/33ab3918-67b0-4b28-9d37-f29e2bb7c5aa" />



<img width="1333" height="663" alt="image" src="https://github.com/user-attachments/assets/3ecaab89-5468-4dc2-ba92-b593bd58a455" />



<img width="1328" height="661" alt="image" src="https://github.com/user-attachments/assets/00608347-75d5-4170-8c66-9e33e0efc167" />



<img width="1226" height="632" alt="image" src="https://github.com/user-attachments/assets/94dcb338-c6c4-4749-8ece-73537db42275" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
