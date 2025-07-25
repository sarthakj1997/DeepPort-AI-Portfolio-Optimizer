# src/data_sourcing.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text, exc
from src.database import get_engine, create_tables, get_session, AssetData

class DataSourcing:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        try:
            # Initialize database connection
            self.engine = get_engine()
            create_tables(self.engine)
            self.session = get_session(self.engine)
            print("Database connection established successfully")
        except exc.SQLAlchemyError as e:
            print(f"Database connection error: {e}")
            raise
    
    def download_and_store_data(self):
        for ticker in self.tickers:
            try:
                print(f"Downloading data for {ticker}")
                raw = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=False,
                    threads=True
                )
                if raw.empty:
                    print(f"No data found for {ticker}")
                    continue

                # Handle the MultiIndex structure from yfinance
                print(f"Downloaded {len(raw)} rows for {ticker}")
                
                # Convert MultiIndex DataFrame to a regular DataFrame
                # First, reset the index to get Date as a column
                raw = raw.reset_index()
                
                # Create a new DataFrame with flattened column names
                flat_data = pd.DataFrame()
                flat_data['date'] = raw[('Date', '')]
                flat_data['ticker'] = ticker
                flat_data['open'] = raw[('Open', ticker)]
                flat_data['high'] = raw[('High', ticker)]
                flat_data['low'] = raw[('Low', ticker)]
                flat_data['close'] = raw[('Close', ticker)]
                flat_data['adj_close'] = raw[('Adj Close', ticker)]
                flat_data['volume'] = raw[('Volume', ticker)]
                
                print(f"Flattened data columns: {flat_data.columns}")
                
                # Process each row individually
                for i in range(len(flat_data)):
                    try:
                        # Extract values as Python native types
                        date_val = flat_data.iloc[i]['date'].to_pydatetime()
                        open_val = float(flat_data.iloc[i]['open'])
                        high_val = float(flat_data.iloc[i]['high'])
                        low_val = float(flat_data.iloc[i]['low'])
                        close_val = float(flat_data.iloc[i]['close'])
                        adj_close_val = float(flat_data.iloc[i]['adj_close'])
                        volume_val = int(flat_data.iloc[i]['volume'])
                        
                        # Create AssetData object with scalar values
                        asset_data = AssetData(
                            ticker=ticker,
                            date=date_val,
                            open=open_val,
                            high=high_val,
                            low=low_val,
                            close=close_val,
                            adj_close=adj_close_val,
                            volume=volume_val
                        )
                        
                        # Add to session and commit immediately
                        self.session.add(asset_data)
                        self.session.commit()
                        
                    except Exception as e:
                        self.session.rollback()
                        print(f"Error processing row {i} for {ticker}: {e}")
                        # Print the problematic row for debugging
                        print(f"Problematic row: {raw.iloc[i]}")
                        continue
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                self.session.rollback()
        
        try:
            self.session.close()
            print("Database session closed successfully")
        except Exception as e:
            print(f"Error closing database session: {e}")