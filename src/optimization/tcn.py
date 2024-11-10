import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, Lambda
from tensorflow.keras.models import Model
from tcn import TCN
import keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Custom Sharpe Ratio Loss Layer
class SharpeRatioLossLayer(Layer):
    def __init__(self, risk_free_rate=0.01, diversification_penalty=0.01, **kwargs):
        super(SharpeRatioLossLayer, self).__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.diversification_penalty = diversification_penalty

    def call(self, inputs):
        y_pred, historical_returns = inputs  # y_pred: predicted weights, historical_returns: asset returns

        # Expand dimensions of y_pred to match the shape of historical_returns
        y_pred_expanded = tf.expand_dims(y_pred, axis=1)
        y_pred_expanded = tf.tile(y_pred_expanded, [1, historical_returns.shape[1], 1])

        # Calculate portfolio returns
        portfolio_returns = tf.reduce_sum(y_pred_expanded * historical_returns, axis=-1)

        # Calculate expected return and portfolio volatility
        expected_return = tf.reduce_mean(portfolio_returns, axis=-1)
        excess_return = expected_return - self.risk_free_rate
        portfolio_volatility = tf.math.reduce_std(portfolio_returns, axis=-1)
        sharpe_ratio = excess_return / (portfolio_volatility + K.epsilon())

        # Diversification penalty
        diversification_penalty = self.diversification_penalty * tf.reduce_sum(tf.square(y_pred), axis=-1)

        # Total loss
        loss = -tf.reduce_mean(sharpe_ratio) + tf.reduce_mean(diversification_penalty)
        
        self.add_loss(loss)
        return y_pred  # Return y_pred as we need it as the output for the model

# Function to build and compile the TCN model
def build_tcn_sharpe_ratio_model(look_back_window, n_features, n_assets):
    # Input layer
    historical_returns_input = Input(shape=(look_back_window, n_features))
    
    # TCN model definition
    tcn_output = TCN(
        nb_filters=32,
        kernel_size=4,
        nb_stacks=3,
        padding='causal',
        dropout_rate=0.2,
        return_sequences=False
    )(historical_returns_input)
    
    # Dense layers for feature extraction
    dense_output = Dense(64, activation='relu')(tcn_output)
    
    # Output layer with normalization
    raw_weights = Dense(n_assets, activation='sigmoid')(dense_output)
    y_pred = Lambda(lambda x: x / tf.reduce_sum(x, axis=1, keepdims=True))(raw_weights)
    
    # Custom Sharpe Ratio loss layer
    sharpe_ratio_layer = SharpeRatioLossLayer()([y_pred, historical_returns_input])
    
    # Create and compile the model
    model = Model(inputs=historical_returns_input, outputs=sharpe_ratio_layer)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=None)
    
    return model

# Function to train the model
def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=50):
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr]
    )
    model_dir='models/tcn/'
    model.save(f"{model_dir}tcn")
    return history


def create_sequence_data(data, look_back_window=30):
        # Create lists to hold the inputs and outputs
    X = []  # Feature set
    y = []  # Target weights

    # Assuming `optimal_weights` is a list of dictionaries containing weights over time
    for i in range(len(data) - look_back_window):
        # Get the features for the current window (look-back period)
        window_data = data.iloc[i: i + look_back_window]
        
        # Convert window_data to a NumPy array and add it to the input list
        X.append(window_data.to_numpy())
    X = np.array(X)
    n_assets = X.shape[2]
    y = np.ones((X.shape[0],n_assets))/n_assets    
    return X,y

def scale_and_split_data(X,y,test_size=0.1):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.08, shuffle=False)
    
    # Flatten X_train and X_val to apply normalization on all features
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    
    # Normalize using StandardScaler (fit on training data and transform on both training and validation)
    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler.transform(X_val_flat)

    # Reshape back to original dimensions
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
    
    return X_train_scaled, X_val_scaled,y_train,y_val