import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.optimize import minimize
import statsmodels.api as sm

from personal_env import PersonalStockEnv


def prepare_data(stock_data, ff_data):
    # Prepare stock data
    stock_data['returns'] = stock_data.groupby('stock_ticker')[
        'prc'].pct_change()
    stock_data = stock_data.pivot(index='date', columns='stock_ticker', values=[
                                  'returns', 'market_cap'])
    stock_data = stock_data.fillna(0)

    # Align Fama-French data
    ff_data = ff_data.reindex(stock_data.index)
    ff_data = ff_data.fillna(method='ffill')

    return stock_data, ff_data


def train_model(env, total_timesteps=100000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_model(model, env, episodes=10):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")


if __name__ == "__main__":
    # Load and prepare data
    stock_data = pd.read_parquet('hackathon_sample_v2.parquet')
    ff_data = pd.read_csv('data/F-F_Research_Data_5_Factors.csv',
                          index_col='Date', parse_dates=True)

    stock_data, ff_data = prepare_data(stock_data, ff_data)

    # Create and wrap the environment
    env = DummyVecEnv([lambda: PersonalStockEnv(stock_data, ff_data)])

    # Train the model
    model = train_model(env)

    # Evaluate the model
    evaluate_model(model, env)
