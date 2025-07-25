import pandas as pd
import numpy as np
import os
import yfinance as yf
from datetime import datetime, timedelta

def process_and_save_data(tickers, start_date, end_date, progress_callback=None):
    """
    Download, process, and save stock data for the given tickers and date range.
    
    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        progress_callback (callable): Function to call with progress updates
    
    Returns:
        dict: Dictionary with success/failure information
    """
    results = {
        "success": [],
        "failed": [],
        "message": ""
    }
    
    # Create directories if they don't exist
    os.makedirs("data/raw_data", exist_ok=True)
    os.makedirs("data/processed_data/train", exist_ok=True)
    os.makedirs("data/processed_data/test", exist_ok=True)
    
    # Calculate the train/test split date (80% train, 20% test)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    total_days = (end_dt - start_dt).days
    train_days = int(total_days * 0.8)
    split_date = start_dt + timedelta(days=train_days)
    split_date_str = split_date.strftime('%Y-%m-%d')
    
    total_tickers = len(tickers)
    
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(f"Processing {ticker} ({i+1}/{total_tickers})", (i / total_tickers) * 100)
        
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                results["failed"].append(ticker)
                continue
                
            # Save raw data
            data.to_csv(f"data/raw_data/{ticker}_raw.csv")
            
            # Process data
            processed_data = data.copy()
            
            # Calculate returns
            processed_data['Returns'] = processed_data['Adj Close'].pct_change()
            processed_data['Log_Returns'] = np.log(processed_data['Adj Close'] / processed_data['Adj Close'].shift(1))
            
            # Calculate moving averages
            processed_data['MA_5'] = processed_data['Adj Close'].rolling(window=5).mean()
            processed_data['MA_20'] = processed_data['Adj Close'].rolling(window=20).mean()
            
            # Calculate volatility
            processed_data['Volatility'] = processed_data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Drop NaN values
            processed_data = processed_data.dropna()
            
            # Split into train and test sets
            train_data = processed_data[:split_date_str]
            test_data = processed_data[split_date_str:]
            
            # Save processed data
            train_data.to_csv(f"data/processed_data/train/{ticker}_processed.csv")
            test_data.to_csv(f"data/processed_data/test/{ticker}_processed.csv")
            
            results["success"].append(ticker)
            
        except Exception as e:
            results["failed"].append(ticker)
            results["message"] += f"Error processing {ticker}: {str(e)}\n"
    
    if progress_callback:
        progress_callback("Processing complete", 100)
    
    return results