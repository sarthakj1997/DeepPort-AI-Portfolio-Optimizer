# src/data_sourcing.py

import yfinance as yf
from datetime import datetime
from src.database import get_engine, create_tables, get_session, AssetData

class DataSourcing:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.engine = get_engine()
        create_tables(self.engine)
        self.session = get_session(self.engine)
    
    def download_and_store_data(self):
        for ticker in self.tickers:
            print(f"Downloading data for {ticker}")
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            df.reset_index(inplace=True)
            for _, row in df.iterrows():
                asset_data = AssetData(
                    ticker=ticker,
                    date=row['Date'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    adj_close=row['Adj Close'],
                    volume=row['Volume']
                )
                self.session.add(asset_data)
            self.session.commit()
