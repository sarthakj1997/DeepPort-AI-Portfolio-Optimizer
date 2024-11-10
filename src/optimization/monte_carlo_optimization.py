import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

def monte_carlo_simulation(log_returns, num_simulations=5000):
    num_assets = len(log_returns.columns)
    results = np.zeros((num_simulations, 3))
    weight_records = []

    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        exp_ret = np.sum((log_returns.mean() * weights) * 252)
        exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
        sharpe_ratio = exp_ret / exp_vol
        
        results[i, 0] = exp_ret
        results[i, 1] = exp_vol
        results[i, 2] = sharpe_ratio
        weight_records.append(weights)

    results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe Ratio'])
    weights_df = pd.DataFrame(weight_records, columns=log_returns.columns)

    return results_df, weights_df

def simulated_annealing(log_returns, initial_weights=None, num_iterations=5000, initial_temp=1.0, cooling_rate=0.995):
    """
    Perform portfolio optimization using Simulated Annealing to maximize the Sharpe Ratio.
    Track and store each intermediate result.

    Parameters:
    - log_returns (pd.DataFrame): DataFrame containing log returns of stocks.
    - initial_weights (np.array): Initial weights for the portfolio. If None, random weights are generated.
    - num_iterations (int): Number of iterations for the simulated annealing process.
    - initial_temp (float): Starting temperature for simulated annealing.
    - cooling_rate (float): Rate at which the temperature decreases.

    Returns:
    - results_df (pd.DataFrame): DataFrame with columns for returns, volatility, Sharpe Ratio, and weights at each iteration.
    """
    num_assets = len(log_returns.columns)
    if initial_weights is None:
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
    else:
        weights = initial_weights

    results = []
    best_sharpe_ratio = -np.inf
    best_result = {}
    current_temp = initial_temp

    for i in range(num_iterations):
        # Calculate portfolio metrics for the current weights
        exp_ret = np.sum((log_returns.mean() * weights) * 252)
        exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
        sharpe_ratio = exp_ret / exp_vol

        # Record the current result
        results.append({
            'Return': exp_ret,
            'Volatility': exp_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Weights': weights.copy()
        })
        
        # Check if the current result is the best so far
        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            best_result = {
                'Return': exp_ret,
                'Volatility': exp_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Weights': weights.copy()
            }

        # Generate new weights by slightly modifying the current weights
        new_weights = weights + np.random.normal(0, 0.05, num_assets)
        new_weights = np.abs(new_weights)
        new_weights /= np.sum(new_weights)

        # Calculate portfolio metrics for the new weights
        new_exp_ret = np.sum((log_returns.mean() * new_weights) * 252)
        new_exp_vol = np.sqrt(np.dot(new_weights.T, np.dot(log_returns.cov() * 252, new_weights)))
        new_sharpe_ratio = new_exp_ret / new_exp_vol

        # Acceptance criterion: accept if new Sharpe Ratio is better or with probability based on temperature
        if (new_sharpe_ratio > sharpe_ratio) or (np.random.rand() < np.exp((new_sharpe_ratio - sharpe_ratio) / current_temp)):
            weights = new_weights

        # Cool down the temperature
        current_temp *= cooling_rate

    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results)
    results_df['Weights'] = results_df['Weights'].apply(lambda w: np.array(w))  # Convert weights to array format

    return results_df, best_result

def plot_monte_carlo_results(results_df):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        results_df['Volatility'], results_df['Return'], 
        c=results_df['Sharpe Ratio'], cmap='viridis', marker='o', alpha=0.6
    )
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Monte Carlo Simulation: Portfolio Optimization')
    plt.grid(True)
    plt.show()

def plot_both(results_df_mc, results_df_sa, best_result_mc, best_result_sa):
    """
    Plot Random Monte Carlo and Simulated Annealing results side by side for comparison.
    Highlight the best result found by each method.

    Parameters:
    - results_df_mc (pd.DataFrame): DataFrame with Random Monte Carlo simulation results.
    - results_df_sa (pd.DataFrame): DataFrame with Simulated Annealing results.
    - best_result_mc (dict): Dictionary with the best portfolio from Monte Carlo simulation.
    - best_result_sa (dict): Dictionary with the best portfolio from Simulated Annealing.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Monte Carlo plot
    scatter_mc = ax1.scatter(
        results_df_mc['Volatility'], results_df_mc['Return'],
        c=results_df_mc['Sharpe Ratio'], cmap='viridis', marker='o', alpha=0.6
    )
    ax1.set_title('Monte Carlo Simulation')
    ax1.set_xlabel('Expected Volatility')
    ax1.set_ylabel('Expected Return')
    ax1.grid(True)
    plt.colorbar(scatter_mc, ax=ax1, label='Sharpe Ratio (Monte Carlo)')

    # Highlight the best result from Monte Carlo
    ax1.scatter(
        best_result_mc['Volatility'], best_result_mc['Return'],
        color='green', marker='*', s=200, label='Best (Monte Carlo)'
    )
    ax1.legend()

    # Simulated Annealing plot
    scatter_sa = ax2.scatter(
        results_df_sa['Volatility'], results_df_sa['Return'],
        c=results_df_sa['Sharpe Ratio'], cmap='plasma', marker='o', alpha=0.6
    )
    ax2.set_title('Simulated Annealing')
    ax2.set_xlabel('Expected Volatility')
    ax2.grid(True)
    plt.colorbar(scatter_sa, ax=ax2, label='Sharpe Ratio (Simulated Annealing)')

    # Highlight the best result from Simulated Annealing
    ax2.scatter(
        best_result_sa['Volatility'], best_result_sa['Return'],
        color='red', marker='*', s=200, label='Best (Simulated Annealing)'
    )
    ax2.legend()

    plt.suptitle('Comparison of Portfolio Optimization: Monte Carlo vs Simulated Annealing')
    plt.show()



