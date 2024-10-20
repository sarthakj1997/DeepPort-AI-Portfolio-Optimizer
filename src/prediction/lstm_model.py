# src/prediction/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import os

class LSTMPredictor:
    def __init__(self, data_dir='data/processed_data/', model_dir='models/lstm/'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_data(self, ticker):
        df = pd.read_csv(f"{self.data_dir}{ticker}_processed.csv", index_col='date', parse_dates=True)
        return df
    
    def preprocess_data(self, df):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df['adj_close'].values.reshape(-1,1))
        return scaled_data, scaler
    
    def create_dataset(self, data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def train_model(self, ticker, epochs=10, batch_size=64):
        df = self.load_data(ticker)
        data, scaler = self.preprocess_data(df)
        X, y = self.create_dataset(data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        model = self.build_model((X.shape[1], 1))
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(f"{self.model_dir}{ticker}_lstm.h5")
        return model, scaler
