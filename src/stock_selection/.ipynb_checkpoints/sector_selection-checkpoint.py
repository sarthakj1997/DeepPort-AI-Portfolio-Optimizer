# stock_selection/sector_selection.py

import yfinance as yf
import pandas as pd
import numpy as np
import os

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
    """
    Calculates the annualized return of a stock based on historical adjusted close prices over a custom period.

    Parameters:
    - ticker (str): Stock ticker.
    - start_date (str): Start date for the historical data.
    - end_date (str): End date for the historical data.

    Returns:
    - float: Annualized return, or None if data could not be fetched.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
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
                            key_stocks=None, start_date="2013-01-01", end_date="2023-01-01"):
    """
    Selects the top-performing stocks from each sector over a specified period, 
    adds key stocks, and saves the tickers to a file.

    Parameters:
    - stocks_per_sector (int): Number of top-performing stocks to select from each sector.
    - ticker_file (str): Path where the selected tickers will be saved.
    - key_stocks (list): List of key stocks to include regardless of sector performance.
    - start_date (str): Start date for the historical data.
    - end_date (str): End date for the historical data.

    Returns:
    - list: List of selected stock tickers.
    """
    sp500_sectors = fetch_sp500_sectors()
    selected_stocks = set()

    # Add key stocks to ensure they're included
    if key_stocks:
        selected_stocks.update(key_stocks)

    # Select top-performing stocks from each sector
    for sector in sp500_sectors['Sector'].unique():
        sector_stocks = sp500_sectors[sp500_sectors['Sector'] == sector]['Ticker']
        returns = {}
        
        # Calculate annualized return for each stock in the sector with longer date range
        for ticker in sector_stocks:
            annualized_return = calculate_annualized_return(ticker, start_date=start_date, end_date=end_date)
            if annualized_return is not None:
                returns[ticker] = annualized_return

        # Select top-performing stocks based on returns
        top_stocks = sorted(returns, key=returns.get, reverse=True)[:stocks_per_sector]
        selected_stocks.update(top_stocks)

    # Save selected tickers to a file
    with open(ticker_file, "w") as f:
        for ticker in selected_stocks:
            f.write(f"{ticker}\n")

    return list(selected_stocks)

