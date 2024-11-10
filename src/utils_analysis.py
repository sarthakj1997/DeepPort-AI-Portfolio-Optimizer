# portfolio_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_portfolio_values(data_dict, allocations, start_amount=10000):
    portfolio_values = {}
    for ticker, df in data_dict.items():
        allocation_amount = start_amount * allocations[ticker]
        portfolio_values[ticker] = df['Normed Return'] * allocation_amount
    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df['Total Portfolio Value'] = portfolio_df.sum(axis=1)
    portfolio_df['Total Daily Return'] = portfolio_df['Total Portfolio Value'].pct_change().fillna(0)
    return portfolio_df
    
def calculate_log_returns(data_dict):
    """
    Calculate log returns for each ticker in data_dict.

    Parameters:
    - data_dict (dict): Dictionary of DataFrames, each containing 'Adj Close' prices for a ticker.

    Returns:
    - log_returns_df (pd.DataFrame): DataFrame with log returns for each ticker, with Date as the index.
    """
    log_returns = {}
    for ticker, df in data_dict.items():
        log_returns[ticker] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    log_returns_df = pd.DataFrame(log_returns).dropna()  # Combine into a DataFrame and drop NaN values
    return log_returns_df

# def plot_daily_returns_histogram(portfolio_df):
#     plt.figure(figsize=(10, 6))
#     plt.hist(portfolio_df['Total Daily Return'], bins=50, color='skyblue', edgecolor='black')
#     plt.title("Histogram of Total Daily Return")
#     plt.xlabel("Daily Return (%)")
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.show()

def calculate_cumulative_return(portfolio_df):
    initial_value = portfolio_df['Total Portfolio Value'].iloc[0]
    final_value = portfolio_df['Total Portfolio Value'].iloc[-1]
    cumulative_return = (final_value / initial_value) - 1
    return cumulative_return

def calculate_sharpe_ratio(portfolio_df, risk_free_rate=0.01):
    risk_free_rate_daily = risk_free_rate / 252
    average_daily_return = portfolio_df['Total Daily Return'].mean()
    std_dev_daily_return = portfolio_df['Total Daily Return'].std()
    sharpe_ratio = (average_daily_return - risk_free_rate_daily) / std_dev_daily_return
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)
    return sharpe_ratio, annualized_sharpe_ratio

def calculate_daily_mean_std(portfolio_df):
    mean_return = portfolio_df['Total Daily Return'].mean()
    std_dev = portfolio_df['Total Daily Return'].std()
    return mean_return, std_dev

# def plot_portfolio_value_and_daily_return(portfolio_df):
#     fig, ax1 = plt.subplots(figsize=(12, 8))
#     ax1.plot(portfolio_df.index, portfolio_df['Total Portfolio Value'], label='Total Portfolio Value', color='blue', linewidth=2)
#     ax1.set_xlabel("Date")
#     ax1.set_ylabel("Total Portfolio Value (in $)", color='blue')
#     ax1.tick_params(axis='y', labelcolor='blue')
#     ax1.legend(loc="upper left")
#     ax2 = ax1.twinx()
#     ax2.plot(portfolio_df.index, portfolio_df['Total Daily Return'], label='Total Daily Return', color='red', linestyle='--', linewidth=1)
#     ax2.set_ylabel("Total Daily Return (%)", color='red')
#     ax2.tick_params(axis='y', labelcolor='red')
#     ax2.legend(loc="upper right")
#     plt.title("Total Portfolio Value and Daily Return Over Time")
#     plt.grid(True)
#     plt.show()

# def plot_individual_ticker_values(portfolio_df, tickers):
#     plt.figure(figsize=(12, 8))
#     for ticker in tickers:
#         plt.plot(portfolio_df.index, portfolio_df[ticker], label=f"{ticker} Portfolio Value")
#     plt.plot(portfolio_df.index, portfolio_df['Total Portfolio Value'], label='Total Portfolio Value', linewidth=2, color='black')
#     plt.title("Portfolio Value Over Time by Ticker")
#     plt.xlabel("Date")
#     plt.ylabel("Portfolio Value (in $)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot_portfolio_value_and_daily_return(portfolio_df, ax1):
    # Plot total portfolio value on the primary y-axis
    ax1.plot(portfolio_df.index, portfolio_df['Total Portfolio Value'], label='Total Portfolio Value', color='blue', linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Portfolio Value (in $)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc="upper left")

    # Create a secondary y-axis for daily returns
    ax2 = ax1.twinx()
    ax2.plot(portfolio_df.index, portfolio_df['Total Daily Return'], label='Total Daily Return', color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel("Total Daily Return (%)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc="upper right")
    
    ax1.set_title("Total Portfolio Value and Daily Return Over Time")
    ax1.grid(True)


def plot_individual_ticker_values(portfolio_df, tickers, ax):
    for ticker in tickers:
        ax.plot(portfolio_df.index, portfolio_df[ticker], label=f"{ticker} Portfolio Value")
    ax.plot(portfolio_df.index, portfolio_df['Total Portfolio Value'], label='Total Portfolio Value', linewidth=2, color='black')
    ax.set_title("Portfolio Value Over Time by Ticker")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (in $)")
    ax.legend()
    ax.grid(True)


def plot_daily_returns_histogram(portfolio_df, ax):
    ax.hist(portfolio_df['Total Daily Return'], bins=50, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Total Daily Return")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.grid(True)


def plot_all_result(allocations,data_dict,start_amount=10000):
    # Calculate portfolio values based on test data
    portfolio_df = calculate_portfolio_values(data_dict, allocations, start_amount)
    tickers = list(data_dict.keys())
    
    # Create a figure and axes for side-by-side plots in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot histogram of daily returns
    plot_daily_returns_histogram(portfolio_df, ax=axs[0, 0])
    
    # Plot total portfolio value and daily return
    plot_portfolio_value_and_daily_return(portfolio_df, ax1=axs[0, 1])

    # Calculate and print cumulative return
    cumulative_return = calculate_cumulative_return(portfolio_df)
    print(f"Cumulative Portfolio Return: {cumulative_return:.2%}")

    # Calculate and print Sharpe Ratio and Annualized Sharpe Ratio
    sharpe_ratio, annualized_sharpe_ratio = calculate_sharpe_ratio(portfolio_df)
    print(f"Daily Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe_ratio:.4f}")

    # Calculate and print mean and standard deviation of daily returns
    mean_return, std_dev = calculate_daily_mean_std(portfolio_df)
    print(f"Mean of Total Daily Return: {mean_return:.4%}")
    print(f"Standard Deviation of Total Daily Return: {std_dev:.4%}")

    # Plot individual ticker values over time
    plot_individual_ticker_values(portfolio_df, tickers, ax=axs[1, 0])
    
    # Leave the last plot blank or add another visualization if desired
    axs[1, 1].axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
