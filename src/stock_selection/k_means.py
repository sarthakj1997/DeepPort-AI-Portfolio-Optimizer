# stock_selection/k_means.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
import os

def calculate_returns_volatility(tickers, start_date="2013-01-01", end_date="2023-01-01"):
    """
    Calculate the annualized return and volatility for each stock in the tickers list.

    Parameters:
    - tickers (list): List of stock tickers.
    - start_date (str): Start date for historical data.
    - end_date (str): End date for historical data.

    Returns:
    - pd.DataFrame: DataFrame with columns for annualized return and volatility for each stock.
    """
    metrics = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, 
                           progress=False, auto_adjust=False, threads=True)
        if data.empty:
            continue
        if 'Adj Close' in data.columns:
            data['Daily Return'] = data['Adj Close'].pct_change()
        else:
            print(f"{ticker}: 'Adj Close' column not found. Using Close")
            data['Daily Return'] = data['Close'].pct_change()
            continue
        
        # Calculate annualized return and volatility
        annualized_return = data['Daily Return'].mean() * 252
        annualized_volatility = data['Daily Return'].std() * np.sqrt(252)
        metrics[ticker] = [annualized_return, annualized_volatility]
    
    metrics_df = pd.DataFrame(metrics, index=['Return', 'Volatility']).T
    return metrics_df

def perform_k_means_clustering(metrics_df, n_clusters=5):
    """
    Perform K-Means clustering on stock metrics (return and volatility) and assign each stock to a cluster.

    Parameters:
    - metrics_df (pd.DataFrame): DataFrame with columns for 'Return' and 'Volatility' for each stock.
    - n_clusters (int): Number of clusters to form.

    Returns:
    - pd.DataFrame: Original metrics DataFrame with an additional 'Cluster' column indicating the cluster assignment.
    """
    scaler = StandardScaler()
    metrics_scaled = scaler.fit_transform(metrics_df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    metrics_df['Cluster'] = kmeans.fit_predict(metrics_scaled)
    
    return metrics_df

def select_representative_stocks(metrics_df, stocks_per_cluster=2, output_file="data/selected_cluster_stocks.txt"):
    """
    Select a specified number of representative stocks from each cluster based on proximity to the cluster centroid.

    Parameters:
    - metrics_df (pd.DataFrame): DataFrame with stock metrics and assigned clusters.
    - stocks_per_cluster (int): Number of stocks to select from each cluster.

    Returns:
    - list: List of tickers representing each cluster.
    """
    representative_stocks = []
    for cluster in metrics_df['Cluster'].unique():
        cluster_stocks = metrics_df[metrics_df['Cluster'] == cluster]
        centroid = cluster_stocks[['Return', 'Volatility']].mean().values
        
        # Calculate distance from each stock to the centroid
        cluster_stocks['Distance'] = cluster_stocks.apply(
            lambda row: np.linalg.norm(row[['Return', 'Volatility']].values - centroid), axis=1
        )
        
        # Select the top stocks_per_cluster closest to the centroid
        closest_stocks = cluster_stocks.nsmallest(stocks_per_cluster, 'Distance').index.tolist()
        representative_stocks.extend(closest_stocks)
        
    # Save selected stocks to a file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for ticker in representative_stocks:
            f.write(f"{ticker}\n")
    
    return representative_stocks


def plot_k_means_clusters(metrics_df, selected_stocks):
    """
    Plot the K-Means clustering results with each stock's return and volatility, 
    highlighting the selected representative stocks.

    Parameters:
    - metrics_df (pd.DataFrame): DataFrame with stock metrics and assigned clusters.
    - selected_stocks (list): List of representative stocks chosen from each cluster.
    """
    plt.figure(figsize=(12, 8))

    # Plot each cluster with a different color
    clusters = metrics_df['Cluster'].unique()
    for cluster in clusters:
        cluster_data = metrics_df[metrics_df['Cluster'] == cluster]
        plt.scatter(cluster_data['Volatility'], cluster_data['Return'], label=f"Cluster {cluster}")

    # Highlight the selected stocks with a distinct marker and label
    selected_data = metrics_df.loc[selected_stocks]
    plt.scatter(selected_data['Volatility'], selected_data['Return'], 
                color='black', marker='X', s=100, label="Selected Stocks")

    # Add labels for selected stocks
    for ticker, row in selected_data.iterrows():
        plt.text(row['Volatility'], row['Return'], ticker, fontsize=9, ha='right')

    # Plot aesthetics
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("K-Means Clustering of Stocks Based on Return and Volatility")
    plt.legend()
    plt.show()


