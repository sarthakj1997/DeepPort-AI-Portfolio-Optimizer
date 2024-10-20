# src/optimization/dqn.py

import gym
import numpy as np
from stable_baselines3 import DQN
from src.database import get_engine, create_tables, get_session, OptimizationResult

class PortfolioEnv(gym.Env):
    def __init__(self, returns):
        super(PortfolioEnv, self).__init__()
        self.returns = returns
        self.num_assets = returns.shape[1]
        self.action_space = gym.spaces.Discrete(self.num_assets)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets,), dtype=np.float32)
        self.current_step = 0
    
    def reset(self):
        self.current_step = 0
        return self.returns.iloc[self.current_step].values
    
    def step(self, action):
        reward = self.returns.iloc[self.current_step, action]
        self.current_step += 1
        done = self.current_step >= len(self.returns)
        obs = self.returns.iloc[self.current_step].values if not done else None
        return obs, reward, done, {}
    
def optimize_with_dqn(returns, tickers):
    env = PortfolioEnv(returns)
    model = DQN('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=10000)
    obs = env.reset()
    done = False
    weights = np.zeros(env.num_assets)
    while not done:
        action, _states = model.predict(obs)
        weights[action] += 1
        obs, reward, done, info = env.step(action)
    weights /= np.sum(weights)
    store_optimization_results('DQN', tickers, weights)
    return weights

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
