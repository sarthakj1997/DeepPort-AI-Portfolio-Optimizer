import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def load_weights():
    """Load portfolio weights from JSON files"""
    weights_dir = 'data/weights'
    weights = {}
    for filename in os.listdir(weights_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(weights_dir, filename), 'r') as f:
                    weights[filename.split('.')[0]] = json.load(f)
            except json.JSONDecodeError as e:
                st.warning(f"Error loading {filename}: {e}")
                # Create a default equal weight portfolio for this algorithm
                with open("data/selected_cluster_stocks.txt", "r") as f:
                    tickers = [line.strip() for line in f]
                equal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
                weights[filename.split('.')[0]] = equal_weights
    return weights

def load_test_data():
    """Load test data for all tickers and S&P 500 benchmark"""
    with open("data/selected_cluster_stocks.txt", "r") as f:
        tickers = [line.strip() for line in f]

    test_file_path_template = "data/processed_data/test/{}_processed.csv"
    test_data_dict = {}

    for ticker in tickers:
        try:
            test_data_dict[ticker] = pd.read_csv(test_file_path_template.format(ticker), index_col='date', parse_dates=True)
        except FileNotFoundError:
            st.warning(f"Could not find data for {ticker}")

    # Download S&P 500 data using yfinance
    try:
        import yfinance as yf
        # Use the same date range as the test data
        all_dates = []
        for df in test_data_dict.values():
            all_dates.extend(df.index)
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
        else:
            # fallback to a default range
            min_date = pd.to_datetime("2022-01-01")
            max_date = pd.to_datetime("2024-01-01")
            
        # Format dates as strings for yfinance
        min_date_str = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)
        max_date_str = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)
        
        st.info(f"Downloading S&P 500 data from {min_date_str} to {max_date_str}")
        
        # Use yfinance download function with string dates
        ticker_obj = yf.Ticker("^GSPC")
        sp500 = ticker_obj.history(start=min_date_str, end=max_date_str)
        
        if not sp500.empty:
            # Use 'Close' if 'Adj Close' is not present
            if 'Adj Close' in sp500.columns:
                sp500_series = sp500['Adj Close'].rename('adj_close')
            elif 'Close' in sp500.columns:
                sp500_series = sp500['Close'].rename('adj_close')
            else:
                st.warning("S&P 500 data does not contain 'Close' or 'Adj Close' column.")
                sp500_series = None
                
            if sp500_series is not None:
                # Store as DataFrame for consistency with other tickers
                # Ensure timezone-naive index
                sp500_series.index = sp500_series.index.tz_localize(None)
                test_data_dict['S&P 500'] = pd.DataFrame(sp500_series)
        else:
            st.warning("Could not download S&P 500 data for benchmark comparison.")
    except Exception as e:
        st.warning(f"Error downloading S&P 500 data: {e}")

    return test_data_dict

def calculate_portfolio_values(weights, test_data_dict, start_amount=10000, start_date=None):
    """Calculate portfolio values over time for each algorithm and S&P 500 benchmark"""
    # Separate S&P 500 from other tickers if present
    test_data_dict = test_data_dict.copy()
    sp500_series = None
    if 'S&P 500' in test_data_dict:
        sp500_series = test_data_dict.pop('S&P 500')['adj_close']

    # Convert test data to a DataFrame where each column is a stock's prices
    test_data_df = pd.DataFrame({ticker: data['adj_close'] for ticker, data in test_data_dict.items()})
    
    # Ensure all datetime indices are timezone-naive
    test_data_df.index = test_data_df.index.tz_localize(None)

    # Filter data based on start_date if provided
    if start_date is not None:
        test_data_df = test_data_df[test_data_df.index >= start_date]
        if sp500_series is not None:
            sp500_series = sp500_series[sp500_series.index >= start_date]
    
    # Normalize test data (convert prices to returns)
    normalized_returns = test_data_df / test_data_df.iloc[0]  # Normalize prices to start at 1

    # Calculate portfolio values over time for each algorithm
    portfolio_values = {}
    for algo, weight_dict in weights.items():
        # Convert weights to a series and ensure the order matches the test data columns
        weight_series = pd.Series(weight_dict).astype(float)
        # Calculate the weighted returns by applying weights to normalized returns
        weighted_returns = normalized_returns.mul(weight_series, axis=1)
        # Sum across all tickers to get the total portfolio value over time
        portfolio_values[algo] = (weighted_returns.sum(axis=1) * start_amount)

    # Add S&P 500 benchmark if available
    if sp500_series is not None:
        # Ensure timezone-naive index
        sp500_series.index = sp500_series.index.tz_localize(None)
        sp500_norm = sp500_series / sp500_series.iloc[0]
        portfolio_values['S&P 500'] = sp500_norm * start_amount

    # Use the index from the test data (or S&P 500 if test data is empty)
    if not test_data_df.empty:
        index = test_data_df.index
    elif sp500_series is not None:
        index = sp500_series.index
    else:
        index = None

    return portfolio_values, index

def app():
    st.set_page_config(page_title="Portfolio Optimization Visualization", layout="wide")
    
    st.title("Portfolio Optimization Visualization")
    
    st.markdown("""
    This application visualizes the performance of different portfolio optimization methods:
    
    - **TCN (Temporal Convolutional Network)**: A deep learning approach using temporal convolutional networks
    - **DQN (Deep Q-Network)**: A reinforcement learning approach using deep Q-networks
    - **MC (Monte Carlo)**: A simulation-based approach using Monte Carlo methods
    - **SA (Simulated Annealing)**: A probabilistic optimization technique
    - **EQ (Equal Weight)**: A baseline strategy with equal weights for all assets
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        weights = load_weights()
        test_data_dict = load_test_data()
        
        if not weights or not test_data_dict:
            st.error("Failed to load data. Please check the data paths.")
            return
        
        # Get available date range
        all_dates = []
        for df in test_data_dict.values():
            all_dates.extend(df.index)
        
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
            
            # Create date picker for investment start date
            st.sidebar.header("Investment Settings")
            start_amount = st.sidebar.number_input("Initial Investment Amount ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
            selected_date = st.sidebar.date_input(
                "Select Investment Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
            
            # Convert selected date to pandas datetime
            selected_date = pd.to_datetime(selected_date)
            
            # Calculate portfolio values with selected date
            portfolio_values, dates = calculate_portfolio_values(weights, test_data_dict, start_amount=start_amount, start_date=selected_date)
        else:
            st.error("No data available for visualization.")
            return
    
    # Display portfolio weights
    st.header("Portfolio Weights by Method")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Portfolio Weights", "Portfolio Metrics"])
    
    with tab1:
        st.subheader("Portfolio Values Over Time")

        # Select which algorithms to display (include S&P 500 if present)
        algos = list(weights.keys())
        if 'S&P 500' in portfolio_values:
            algos.append('S&P 500')
        selected_algos = st.multiselect("Select optimization methods to display (including S&P 500)", algos, default=algos)

        if not selected_algos:
            st.warning("Please select at least one optimization method.")
        else:
            # Create a DataFrame for the selected algorithms
            df_values = pd.DataFrame({algo: values for algo, values in portfolio_values.items() if algo in selected_algos})
            
            # Ensure all indices are timezone-naive
            if hasattr(df_values.index, 'tz_localize'):
                df_values.index = df_values.index.tz_localize(None)

            # Plot the portfolio values
            fig, ax = plt.subplots(figsize=(12, 6))
            for algo in selected_algos:
                ax.plot(df_values.index, df_values[algo], label=algo)

            ax.set_title(f"Portfolio Values Over Time (Initial Investment: ${start_amount:,} on {selected_date.strftime('%Y-%m-%d')})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Display the data as a table
            st.subheader("Portfolio Values Data")
            st.dataframe(df_values)

            # Download button for the data
            csv = df_values.to_csv()
            st.download_button(
                label="Download Portfolio Values as CSV",
                data=csv,
                file_name="portfolio_values.csv",
                mime="text/csv",
            )
    
    with tab2:
        st.subheader("Portfolio Weights Comparison")
        
        # Select which algorithm to display weights for
        selected_algo = st.selectbox("Select optimization method", algos)
        
        if selected_algo:
            # Display weights as a bar chart
            weights_df = pd.DataFrame(weights[selected_algo], index=['Weight']).T
            
            fig, ax = plt.subplots(figsize=(12, 6))
            weights_df.plot(kind='bar', ax=ax)
            ax.set_title(f"Portfolio Weights for {selected_algo}")
            ax.set_ylabel("Weight")
            ax.set_xlabel("Asset")
            ax.grid(True, axis='y')
            st.pyplot(fig)
            
            # Display the weights as a table
            st.dataframe(weights_df)
            
            # Download button for the weights
            csv = weights_df.to_csv()
            st.download_button(
                label=f"Download {selected_algo} Weights as CSV",
                data=csv,
                file_name=f"{selected_algo}_weights.csv",
                mime="text/csv",
            )
    
    with tab3:
        st.subheader("Portfolio Performance Metrics")

        # Calculate performance metrics for each algorithm (including S&P 500 if present)
        metrics = {}
        for algo, values in portfolio_values.items():
            # Convert to pandas series if it's not already
            if not isinstance(values, pd.Series):
                values = pd.Series(values)
                
            # Calculate returns
            returns = values.pct_change().dropna()
            
            # Calculate metrics
            total_return = (values.iloc[-1] / values.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(values)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            max_drawdown = (values / values.cummax() - 1).min()
            
            metrics[algo] = {
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown
            }
        
        # Convert to DataFrame and display
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.sort_values('Sharpe Ratio', ascending=False)
        
        # Format the metrics for display
        formatted_metrics = metrics_df.copy()
        for col in ['Total Return', 'Annual Return', 'Volatility', 'Max Drawdown']:
            formatted_metrics[col] = formatted_metrics[col].map('{:.2%}'.format)
        formatted_metrics['Sharpe Ratio'] = formatted_metrics['Sharpe Ratio'].map('{:.2f}'.format)
        
        st.dataframe(formatted_metrics)
        
        # Plot the metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total Return
        metrics_df['Total Return'].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Total Return')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, axis='y')
        
        # Annual Return
        metrics_df['Annual Return'].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Annual Return')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].grid(True, axis='y')
        
        # Sharpe Ratio
        metrics_df['Sharpe Ratio'].plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].grid(True, axis='y')
        
        # Max Drawdown
        metrics_df['Max Drawdown'].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Max Drawdown')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button for the metrics
        csv = metrics_df.to_csv()
        st.download_button(
            label="Download Performance Metrics as CSV",
            data=csv,
            file_name="portfolio_metrics.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    app()