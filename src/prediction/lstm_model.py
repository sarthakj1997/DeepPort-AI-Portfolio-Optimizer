# src/prediction/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
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
        model.add(LSTM(30, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.25))
        model.add(LSTM(units=20,return_sequences=True))
        model.add(Dropout(0.30))
        model.add(LSTM(units=20,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(units=20))
        model.add(Dropout(0.30))
        model.add(Dense(units=1))
        model.compile(optimizer='adam',loss='mean_squared_error')
        return model
    
    def train_model(self, ticker, epochs=20, batch_size=64):
        df = self.load_data(ticker)
        data, scaler = self.preprocess_data(df)
        train_size = int(len(data)*0.95)
        train_data = data[:train_size]
        X, y = self.create_dataset(data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        model = self.build_model((X.shape[1], 1))
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(f"{self.model_dir}{ticker}_lstm.h5")
        return model, scaler
    
    def test_model(self, ticker, test_size=30):
        # Load and preprocess data
        df = self.load_data(ticker)
        data, scaler = self.preprocess_data(df)
        
        # Prepare training and testing datasets
        train_data = data[:(-test_size-60)]
        test_data = data[(-test_size-60):]
        X_train, y_train = self.create_dataset(train_data)
        X_test, y_test = self.create_dataset(test_data)
        
        print("Shape of X_test:", X_test.shape)  # Should be (samples, time_steps, 1)
        print("Shape of y_test:", y_test.shape)  # Should be (samples, 1)
        

        
        # Load the model
        model = self.build_model((X_train.shape[1], 1))
        model.load_weights(f"{self.model_dir}{ticker}_lstm.h5")
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Reverse scaling
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        return y_test, y_pred
    
    
    def load_trained_model(self, ticker):
        # Load a saved model
        model = self.build_model((60, 1))  # Assuming the time_step used during training is 60
        model.load_weights(f"{self.model_dir}{ticker}_lstm.h5")
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = self.load_data(ticker)
        scaler.fit(df['adj_close'].values.reshape(-1, 1))
        print(f"LSTM model loaded for {ticker}")
        return model, scaler
