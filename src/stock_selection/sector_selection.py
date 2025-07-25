# stock_selection/sector_selection.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def fetch_sp500_sectors():
    """
    Fetches the list of S&P 500 companies with their sectors from Wikipedia.
    Returns a DataFrame with columns 'Ticker' and 'Sector'.
    """
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(sp500_url)
    df = table[0][['Symbol', 'GICS Sector']]  # Get ticker and sector columns
    df.columns = ['Ticker', 'Sector']  # Rename columns for consistency
    return df
    
def calculate_annualized_return(ticker, start_date="2013-01-01", end_date="2023-01-01"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False, threads=True)
        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d") +timedelta(days=1)
        end_date = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=2)
        # Check for full data coverage
        if data.empty or data.index[0].strftime('%Y-%m-%d') != start_date.strftime('%Y-%m-%d') or data.index[-1].strftime('%Y-%m-%d') != end_date.strftime('%Y-%m-%d'):
            print(f'{ticker} does not have data for the whole period')
            return None
        
        data['Daily Return'] = data['Adj Close'].pct_change()
        cumulative_return = (1 + data['Daily Return']).prod() - 1
        trading_days = len(data)
        annualized_return = (1 + cumulative_return) ** (252 / trading_days) - 1
        return annualized_return
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None



def select_stocks_by_sector(stocks_per_sector=3, ticker_file="data/selected_tickers.txt", 
                            key_stocks=None, start_date="2013-01-01", end_date="2023-01-01", 
                            test_start_date="2023-01-01", test_end_date="2024-01-01"):

    sp500_sectors = fetch_sp500_sectors()  # Assuming this function fetches sector data with tickers
    print("S&P 500 sectors fetched successfully.")
    selected_stocks = set()

    # Add key stocks to ensure they're included
    if key_stocks:
        selected_stocks.update(key_stocks)
        print("Key stocks added" )

    for ticker in key_stocks:
        print(f"Key stock: {ticker}")

    # Select top-performing stocks from each sector
    for sector in sp500_sectors['Sector'].unique():
        sector_stocks = sp500_sectors[sp500_sectors['Sector'] == sector]['Ticker']
        returns = {}
        
        # Calculate annualized return for each stock in the sector for the historical period
        for ticker in sector_stocks:
            annualized_return = calculate_annualized_return(ticker, start_date=start_date, end_date=end_date)
            if annualized_return is not None:
                # Check if the stock has complete data for the test period
                try:
                    # Download data for the test period
                    data_test = yf.download(ticker, start=test_start_date, end=test_end_date)

                    # Convert test_start_date and test_end_date to datetime objects
                    test_start_date_dt = datetime.strptime(test_start_date, "%Y-%m-%d")
                    test_end_date_dt = datetime.strptime(test_end_date, "%Y-%m-%d")

                    # Ensure data covers the full test range
                    if (data_test.empty or 
                        not (test_start_date_dt - timedelta(days=3) <= data_test.index[0] <= test_start_date_dt + timedelta(days=3)) or
                        not (test_end_date_dt - timedelta(days=3) <= data_test.index[-1] <= test_end_date_dt + timedelta(days=3))):
                        print(test_start_date_dt)
                        print(data_test.index[0])
                        print(f"{ticker} does not have data for the entire test period.")
                    else:
                        returns[ticker] = annualized_return  # Include in returns only if data is complete

                except Exception:
                    print(f"Error fetching test data for {ticker}. Skipping...")
                    pass  # Continue if there's an error fetching data for the test period

        # Select top-performing stocks based on returns
        top_stocks = sorted(returns, key=returns.get, reverse=True)[:stocks_per_sector]
        selected_stocks.update(top_stocks)

    # Save selected tickers to a file
    with open(ticker_file, "w") as f:
        for ticker in selected_stocks:
            f.write(f"{ticker}\n")

    return list(selected_stocks)
