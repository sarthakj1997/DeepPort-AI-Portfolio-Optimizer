import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
import gym



# Define the custom environment
class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_step = 0
        self.done = False
        self.portfolio_value = initial_balance

        # Actions: represent percentage allocation for each asset
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)

        # State includes asset returns and portfolio value
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(data.columns) + 1,), dtype=np.float32)

    def reset(self):
        self.current_balance = self.initial_balance
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.done = False
        self.state = np.concatenate(([self.portfolio_value], self.data.iloc[self.current_step].values))
        return self.state

    def step(self, action):
        self.current_step += 1
        weights = action / action.sum()
        returns = self.data.iloc[self.current_step].values
        self.portfolio_value = self.portfolio_value * (1 + np.dot(returns, weights))
        reward = self.portfolio_value - self.current_balance  # reward is profit

        self.state = np.concatenate(([self.portfolio_value], returns))
        self.current_balance = self.portfolio_value
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return self.state, reward, self.done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value}")
        


# Define the Double DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Initialize the policy (main) network and target network
        self.model = build_model(state_size, action_size, learning_rate)
        self.target_model = build_model(state_size, action_size, learning_rate)
        self.update_target_network()  # Ensure they start with the same weights

    def update_target_network(self):
        # Copy the weights from the policy network to the target network
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)  # Random action
        return self.model.predict(state, verbose=0)[0]  # Action based on policy network
    
    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        # Sample a minibatch of experiences from the memory
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Use the target network to calculate stable Q-value
                target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][np.argmax(action)] = target  # Update only the chosen action

            # Train the policy network
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            losses.append(history.history['loss'][0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return np.mean(losses) if losses else None  # Return average loss for tracking




# Define the Q-network model
def build_model(input_shape, output_shape, learning_rate=0.001):
    model = Sequential([
        Dense(64, input_dim=input_shape, activation="relu"),
        Dense(64, activation="relu"),
        Dense(output_shape, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model

