# src/data_preprocessing.py

from src.database import get_engine, get_session, AssetData
import pandas as pd
import numpy as np

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
            df = df[~df.index.duplicated(keep='first')]
            data[ticker] = df
        return data
    
    def preprocess_data(self, data):
        processed_data = {}
        for ticker, df in data.items():
            df['Returns'] = df['adj_close'].pct_change().fillna(0)
            df['Normed Return'] = df['adj_close'] / df['adj_close'].iloc[0]
            df['Log Return'] = np.log(df['adj_close'] / df['adj_close'].shift(1)).fillna(0)
            df.dropna(inplace=True)
            processed_data[ticker] = df
        return processed_data

    def split_data(self,data_dict, start_date=None, end_date=None, test_by_date=True, 
                            test_start_date=None, test_end_date=None, test_size=0.1):
        """
        Splits each DataFrame in the dictionary into train and test sets based on either a date range or a test size percentage.

        Parameters:
        data_dict (dict): Dictionary with ticker symbols as keys and DataFrames as values.
        start_date (str): The start date for filtering the data (in 'YYYY-MM-DD' format).
        end_date (str): The end date for filtering the data (in 'YYYY-MM-DD' format).
        interval (str): The data frequency, e.g., "1d" for daily data.
        test_by_date (bool): If True, splits the test set by specified test_start_date and test_end_date.
                            If False, splits the test set based on test_size percentage.
        test_start_date (str, optional): The start date for the test set if test_by_date is True.
        test_end_date (str, optional): The end date for the test set if test_by_date is True.
        test_size (float, optional): The percentage (between 0 and 1) for the test set if test_by_date is False.

        Returns:
        dict: Two dictionaries, train_data_dict and test_data_dict, with ticker symbols as keys and 
            DataFrames as values for the respective train and test sets.
        """
        train_data_dict = {}
        test_data_dict = {}

        for ticker, df in data_dict.items():
            # Ensure the DataFrame is sorted by date
            df = df.sort_index()

            # Filter the DataFrame for the specified date range and interval
            df = df.loc[start_date:end_date]

            if test_by_date:
                # Check if both test_start_date and test_end_date are provided
                if not test_start_date or not test_end_date:
                    raise ValueError("test_start_date and test_end_date must be provided when test_by_date is True.")
                
                # Split by date range for test set
                train_df = df[(df.index < test_start_date) | (df.index > test_end_date)]
                test_df = df[(df.index >= test_start_date) & (df.index <= test_end_date)]
            
            else:
                # Split by test size percentage
                split_point = int(len(df) * (1 - test_size))
                train_df = df.iloc[:split_point]
                test_df = df.iloc[split_point:]
            
            # Store the train and test DataFrames in separate dictionaries
            train_data_dict[ticker] = train_df
            test_data_dict[ticker] = test_df

        return train_data_dict, test_data_dict
