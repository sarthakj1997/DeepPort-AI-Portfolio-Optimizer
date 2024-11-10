# src/metrics.py

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calculate_metrics(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    return mae, mse, rmse
