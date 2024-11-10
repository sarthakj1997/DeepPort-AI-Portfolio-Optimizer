# minimize_sharpe_ratio.py

import numpy as np
from scipy.optimize import minimize

def get_ret_vol_sr(weights, log_returns):
    """
    Calculate the return, volatility, and Sharpe Ratio for a given portfolio allocation.

    Parameters:
    - weights (np.array): Portfolio weights.
    - log_returns (pd.DataFrame): Log returns of assets.

    Returns:
    - np.array: Array containing [return, volatility, Sharpe Ratio].
    """
    weights = np.array(weights)
    ret = np.sum(log_returns.mean() * weights) * 252  # Annualized return
    vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))  # Annualized volatility
    sr = ret / vol  # Sharpe Ratio
    return np.array([ret, vol, sr])

def neg_sharpe(weights, log_returns):
    """
    Negative Sharpe Ratio for optimization.

    Parameters:
    - weights (np.array): Portfolio weights.
    - log_returns (pd.DataFrame): Log returns of assets.

    Returns:
    - float: Negative Sharpe Ratio.
    """
    return get_ret_vol_sr(weights, log_returns)[2] * -1

def check_sum(weights):
    """
    Constraint function to ensure portfolio weights sum to 1.

    Parameters:
    - weights (np.array): Portfolio weights.

    Returns:
    - float: Difference from 1 (should be zero when weights sum to 1).
    """
    return np.sum(weights) - 1

def maximize_sharpe_ratio(log_returns):
    """
    Perform portfolio optimization to maximize the Sharpe Ratio.

    Parameters:
    - log_returns (pd.DataFrame): Log returns of assets.

    Returns:
    - dict: Optimal portfolio weights, expected return, volatility, and Sharpe Ratio.
    """
    num_assets = len(log_returns.columns)
    init_guess = [1 / num_assets for _ in range(num_assets)]
    bounds = tuple((0, 1) for _ in range(num_assets))
    cons = ({'type': 'eq', 'fun': check_sum})

    # Run the optimization
    opt_results = minimize(neg_sharpe, init_guess, args=(log_returns,), method='SLSQP', bounds=bounds, constraints=cons)

    # Retrieve optimal weights and metrics
    optimal_weights = opt_results.x
    opt_ret, opt_vol, opt_sr = get_ret_vol_sr(optimal_weights, log_returns)

    # Return results as a dictionary
    return {
        'Optimal Weights': optimal_weights,
        'Expected Return': opt_ret,
        'Expected Volatility': opt_vol,
        'Sharpe Ratio': opt_sr
    }

