# src/prediction/kalman_filter.py

from pykalman import KalmanFilter
import pandas as pd
import os
import pickle
import numpy as np

class KalmanFilterPredictor:
    def __init__(self, data_dir='data/processed_data/', model_dir='models/kalman_filter/'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self, ticker):
        # Load the CSV file and prepare it for Kalman Filter
        df = pd.read_csv(f"{self.data_dir}{ticker}_processed.csv", index_col='date', parse_dates=True)
        df = df[['adj_close']].reset_index().rename(columns={'date': 'ds', 'adj_close': 'y'})
        return df

    def train_model(self, ticker):
        # Load data
        df = self.load_data(ticker)
        prices = df['y'].values

        # Define a Kalman Filter with both position and velocity
        transition_matrices = [[1, 1], [0, 1]]
        observation_matrices = [[1, 0]]

        # Initialize and fit the Kalman Filter
        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            initial_state_mean=[prices[0], 0],
            n_dim_state=2,
            n_dim_obs=1,
            transition_covariance=np.eye(2) * 0.1,  # Adding some noise to the transition
            observation_covariance=1.0  # Adding some observation noise
        )
        kf = kf.em(prices, n_iter=10)

        # Save the trained Kalman Filter model
        model_file = f"{self.model_dir}{ticker}_kalman_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(kf, f)
        
        print(f"Kalman Filter model trained and saved for {ticker}")
        return kf

    def forecast(self, model, df, steps=30):
        # Create a forecast using the Kalman Filter model
        prices = df['y'].values
        smoothed_state_means, _ = model.filter(prices)

        # Forecast future steps using the dynamic model with noise
        last_state = smoothed_state_means[-1]
        future_forecast = []
        for _ in range(steps):
            # Use the transition matrix to predict the next state with added random noise
            last_state = model.transition_matrices @ last_state + np.random.multivariate_normal([0, 0], model.transition_covariance)
            future_forecast.append(last_state[0])

        future_dates = pd.date_range(df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=steps)
        forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': future_forecast})
        return forecast_df

    def load_trained_model(self, ticker):
        # Load a saved model
        model_file = f"{self.model_dir}{ticker}_kalman_model.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        print(f"Kalman Filter model loaded for {ticker}")
        return model
