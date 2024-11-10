# src/prediction/arima_model.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import os

class ARIMAPredictor:
    def __init__(self, data_dir='data/processed_data/', model_dir='models/arima/'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self, ticker):
    # Load CSV file
        df = pd.read_csv(f"{self.data_dir}{ticker}_processed.csv", index_col='date', parse_dates=True)
        
        # Ensure the index is sorted and drop duplicates if any
        df = df.sort_index()  # Sort index to ensure monotonic order
        df = df[~df.index.duplicated(keep='first')]  # Drop duplicates by keeping the first occurrence

        # Set frequency to business days ('B') if the data is not already in that format
        df = df.asfreq('B')
        
        # Forward fill any missing data after setting frequency (optional, for continuity in ARIMA)
        df.fillna(method='ffill', inplace=True)
        
        return df
    
    def check_stationarity_with_differencing(self, data):
        # First-order differencing
        differenced_data = data.diff().dropna()
        
        # Perform ADF test on differenced data
        result = adfuller(differenced_data)
        print("Differenced ADF Statistic:", result[0])
        print("Differenced p-value:", result[1])
        if result[1] > 0.05:
            print("Differenced series is not stationary.")
        else:
            print("Differenced series is stationary.")
        return differenced_data



    def check_stationarity(self, data):
        # Augmented Dickey-Fuller test to check if the series is stationary
        result = adfuller(data)
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        if result[1] > 0.05:
            print("Series is not stationary.")
            return False
        else:
            print("Series is stationary.")
            return True

    def train_arima(self, ticker, order=(1,1,1)):
        df = self.load_data(ticker)
        
        # Check original data stationarity
        if not self.check_stationarity(df['adj_close']):
            print(f"Data for {ticker} is not stationary. Applying differencing.")
            # Apply differencing and re-check
            differenced_data = self.check_stationarity_with_differencing(df['adj_close'])
        else:
            differenced_data = df['adj_close']
        
        # Train ARIMA with the differenced series or the original if already stationary
        model = ARIMA(differenced_data, order=order)
        model_fit = model.fit()

        # Save the ARIMA model
        model_fit.save(f"{self.model_dir}{ticker}_arima.pkl")
        print(f"ARIMA model trained and saved for {ticker} with order {order}")
        return model_fit


    def train_sarima(self, ticker, order=(1,1,1), seasonal_order=(1,1,1,12)):
        df = self.load_data(ticker)
        if not self.check_stationarity(df['adj_close']):
            print(f"Consider differencing for ticker {ticker} to make the series stationary.")

        # Train SARIMA model for seasonality
        model = SARIMAX(df['adj_close'], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        # Save the SARIMA model
        model_fit.save(f"{self.model_dir}{ticker}_sarima.pkl")
        print(f"SARIMA model trained and saved for {ticker} with order {order} and seasonal order {seasonal_order}")
        return model_fit

    def forecast(self, model_fit, steps=30):
        forecast = model_fit.forecast(steps=steps)
        return forecast

# Example usage:
# predictor = ARIMAPredictor()
# model_fit = predictor.train_arima('AAPL', order=(1, 1, 1))
# forecast = predictor.forecast(model_fit, steps=30)
