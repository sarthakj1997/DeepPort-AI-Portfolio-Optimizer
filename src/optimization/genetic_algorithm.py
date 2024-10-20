# src/optimization/genetic_algorithm.py

from deap import base, creator, tools, algorithms
import numpy as np
from src.database import get_engine, create_tables, get_session, OptimizationResult
import datetime

class GeneticAlgorithmOptimizer:
    def __init__(self, returns, expected_returns, cov_matrix, population_size=100, generations=50):
        self.returns = returns
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.population_size = population_size
        self.generations = generations
        self.engine = get_engine()
        create_tables(self.engine)
        self.session = get_session(self.engine)
    
    def optimize(self):
        num_assets = len(self.expected_returns)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_weight", np.random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n=num_assets)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            weights = np.array(individual)
            weights /= np.sum(weights)
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            fitness = portfolio_return - portfolio_variance  # Adjust as needed
            return (fitness,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=self.population_size)
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=self.generations, verbose=False)
        
        best_individual = tools.selBest(population, k=1)[0]
        best_weights = np.array(best_individual)
        best_weights /= np.sum(best_weights)
        
        self.store_optimization_results('Genetic Algorithm', self.returns.columns.tolist(), best_weights)
        
        return best_weights
    
    def store_optimization_results(self, method, tickers, weights):
        for ticker, weight in zip(tickers, weights):
            result = OptimizationResult(
                method=method,
                ticker=ticker,
                weight=weight
            )
            self.session.add(result)
        self.session.commit()
