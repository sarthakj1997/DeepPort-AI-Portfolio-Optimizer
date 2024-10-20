# src/optimization/classical_methods.py

import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns, CLA, HRPOpt, BlackLittermanModel
from src.database import get_engine, create_tables, get_session, OptimizationResult

class ClassicalMethods:
    def __init__(self, prices):
        self.prices = prices
        self.engine = get_engine()
        create_tables(self.engine)
        self.session = get_session(self.engine)
    
    def mean_variance_optimization(self):
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.sample_cov(self.prices)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        self.store_optimization_results('Mean-Variance', cleaned_weights)
        return cleaned_weights
    
    def critical_line_algorithm(self):
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.sample_cov(self.prices)
        cla = CLA(mu, S)
        weights = cla.max_sharpe()
        self.store_optimization_results('Critical Line Algorithm', weights)
        return weights
    
    def hierarchical_risk_parity(self):
        returns = self.prices.pct_change().dropna()
        cov = returns.cov()
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        self.store_optimization_results('Hierarchical Risk Parity', weights)
        return weights
    
    def black_litterman_model(self, market_prices, prior_returns, Q, P, omega=None):
        S = risk_models.sample_cov(market_prices)
        bl = BlackLittermanModel(S, pi=prior_returns, absolute_views=Q, P=P, omega=omega)
        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        ef = EfficientFrontier(ret_bl, S_bl)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        self.store_optimization_results('Black-Litterman', cleaned_weights)
        return cleaned_weights
    
    def store_optimization_results(self, method, weights):
        for ticker, weight in weights.items():
            result = OptimizationResult(
                method=method,
                ticker=ticker,
                weight=weight
            )
            self.session.add(result)
        self.session.commit()
