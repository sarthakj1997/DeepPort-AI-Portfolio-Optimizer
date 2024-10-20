# src/data_preprocessing.py

from src.database import get_engine, get_session, AssetData
import pandas as pd

class DataPreprocessing:
    def __init__(self):
        self.engine = get_engine()
        self.session = get_session(self.engine)
    
    def load_data(self, tickers):
        data = {}
        for ticker in tickers:
            print(f"Loading data for {ticker}")
            query = self.session.query(AssetData).filter(AssetData.ticker == ticker)
            df = pd.read_sql(query.statement, self.session.bind)
            df.set_index('date', inplace=True)
            data[ticker] = df
        return data
    
    def preprocess_data(self, data):
        processed_data = {}
        for ticker, df in data.items():
            df['Returns'] = df['adj_close'].pct_change()
            df.dropna(inplace=True)
            processed_data[ticker] = df
        return processed_data
