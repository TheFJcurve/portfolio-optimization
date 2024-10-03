
import numpy as np
import gym
from gym import spaces
import statsmodels.api as sm


class PersonalStockEnv(gym.Env):
    def __init__(self, data, ff_data, lookback_period=12, portfolio_size=(50, 100), rebalance_period=21):
        super(PersonalStockEnv, self).__init__()
        self.data = data
        self.ff_data = ff_data
        self.lookback_period = lookback_period
        self.portfolio_size = portfolio_size
        self.rebalance_period = rebalance_period

        self.num_stocks = len(self.data['stock_ticker'].unique())
        self.current_step = lookback_period
        self.last_rebalance_step = self.current_step

        # Action space: portfolio weights for each stock (long/short)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # Observation space: historical returns, BL expected returns, FF factor loadings
        obs_shape = (self.lookback_period * self.num_stocks) + \
            (2 * self.num_stocks)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self):
        self.current_step = self.lookback_period
        self.last_rebalance_step = self.current_step
        return self._next_observation()

    def step(self, action):
        self.current_step += 1

        # Only rebalance at specified intervals
        if self.current_step - self.last_rebalance_step >= self.rebalance_period:
            self.last_rebalance_step = self.current_step
            portfolio_weights = self._get_portfolio_weights(action)
            returns = self.data.loc[self.current_step, 'returns'].values
            reward = np.dot(portfolio_weights, returns)

            done = self.current_step >= len(self.data) - 1
            info = {'portfolio_weights': portfolio_weights}
        else:
            reward = 0
            done = False
            info = {}

        next_obs = self._next_observation()
        return next_obs, reward, done, info

    def _next_observation(self):
        start = self.current_step - self.lookback_period
        end = self.current_step

        historical_returns = self.data.loc[start:end,
                                           'returns'].values.flatten()

        bl_returns = self._get_black_litterman_returns()
        ff_loadings = self._get_fama_french_loadings()

        return np.concatenate([historical_returns, bl_returns, ff_loadings])

    def _get_portfolio_weights(self, action):
        # Convert actions to portfolio weights
        weights = action / np.sum(np.abs(action))

        # Enforce portfolio size constraints
        num_assets = np.random.randint(
            self.portfolio_size[0], self.portfolio_size[1] + 1)
        top_indices = np.argsort(np.abs(weights))[-num_assets:]
        weights[~np.isin(np.arange(len(weights)), top_indices)] = 0

        return weights / np.sum(np.abs(weights))

    def _get_black_litterman_returns(self):
        # Simplified Black-Litterman model
        market_caps = self.data.loc[self.current_step, 'market_cap'].values
        historical_returns = self.data.loc[self.current_step -
                                           self.lookback_period:self.current_step, 'returns']

        # Prior (CAPM)
        risk_aversion = 2.5
        cov_matrix = historical_returns.cov().values
        prior_returns = risk_aversion * \
            np.dot(cov_matrix, market_caps / np.sum(market_caps))

        # Investor views (simplified)
        P = np.eye(self.num_stocks)
        Q = historical_returns.mean().values
        omega = np.diag(np.diag(cov_matrix)) * 0.1

        # Posterior
        tau = 0.05
        A = np.linalg.inv(tau * cov_matrix)
        B = np.dot(P.T, np.linalg.inv(omega))
        C = np.dot(B, P)
        D = np.dot(B, Q)
        posterior_returns = np.dot(np.linalg.inv(
            A + C), np.dot(A, prior_returns) + D)

        return posterior_returns

    def _get_fama_french_loadings(self):
        historical_returns = self.data.loc[self.current_step -
                                           self.lookback_period:self.current_step, 'returns']
        ff_factors = self.ff_data.loc[self.current_step -
                                      self.lookback_period:self.current_step]

        X = sm.add_constant(ff_factors)
        loadings = []

        for stock in historical_returns.columns:
            model = sm.OLS(historical_returns[stock], X).fit()
            loadings.append(model.params)

        return np.array(loadings).flatten()
