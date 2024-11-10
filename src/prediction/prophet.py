# src/prediction/prophet.py

from prophet import Prophet
from prophet.models import StanBackendEnum
import pandas as pd
import os
import pickle

class ProphetPredictor:
    def __init__(self, data_dir='data/processed_data/', model_dir='models/prophet/'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self, ticker):
        # Load the CSV file and prepare it for Prophet
        df = pd.read_csv(f"{self.data_dir}{ticker}_processed.csv", index_col='date', parse_dates=True)
        
        # Prophet expects columns 'ds' for dates and 'y' for values
        df = df[['adj_close']].reset_index().rename(columns={'date': 'ds', 'adj_close': 'y'})
        return df

    def train_model(self, ticker):
        # Load data
        df = self.load_data(ticker)
        
        # Initialize and fit the Prophet model using pystan backend
        model = Prophet(stan_backend='pystan')
        
        # Fit the model
        model.fit(df)

        # Save the model as a pickle file
        model_file = f"{self.model_dir}{ticker}_prophet_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        
        print(f"Prophet model trained and saved for {ticker}")
        return model

    def forecast(self, model, periods=30):
        # Create a DataFrame to hold future dates for forecasting
        future = model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def load_trained_model(self, ticker):
        # Load a saved model
        model_file = f"{self.model_dir}{ticker}_prophet_model.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        print(f"Prophet model loaded for {ticker}")
        return model
