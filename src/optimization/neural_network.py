# src/optimization/neural_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.database import get_engine, create_tables, get_session, OptimizationResult

class NNOptimizer(nn.Module):
    def __init__(self, num_assets):
        super(NNOptimizer, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_assets) / num_assets)
    
    def forward(self):
        weights = torch.softmax(self.weights, dim=0)
        return weights

def optimize_with_nn(expected_returns, cov_matrix, tickers, epochs=1000, lr=0.01):
    num_assets = len(expected_returns)
    model = NNOptimizer(num_assets)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    expected_returns = torch.tensor(expected_returns, dtype=torch.float32)
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        weights = model()
        portfolio_return = torch.dot(weights, expected_returns)
        portfolio_variance = torch.matmul(weights.T, torch.matmul(cov_matrix, weights))
        loss = -portfolio_return + portfolio_variance  # Adjust as needed
        loss.backward()
        optimizer.step()
    
    optimal_weights = model().detach().numpy()
    store_optimization_results('Neural Network', tickers, optimal_weights)
    return optimal_weights

def store_optimization_results(method, tickers, weights):
    engine = get_engine()
    create_tables(engine)
    session = get_session(engine)
    for ticker, weight in zip(tickers, weights):
        result = OptimizationResult(
            method=method,
            ticker=ticker,
            weight=weight
        )
        session.add(result)
    session.commit()
