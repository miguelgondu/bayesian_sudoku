"""
This implements a Gaussian process with the data
passed by the user's interactions.

Taken and adapted from:
https://krasserm.github.io/2018/03/21/bayesian-optimization/
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm

class BayesianOptimization:
    def __init__(self, goal=-10, domain=np.linspace(0, 1, 100)):
        self.domain = domain
        self.kernel = Matern(length_scale=1.0, nu=2.5)
        self.noise = 0.1
        self.goal = goal
        self.X = []
        self.Y = []
        self.gpr = None

    def fit_gpr(self, x=None, y=None):
        if x is not None and y is not None:
            self.X.append(x)
            self.Y.append(y)
        elif x is None and y is None:
            pass
        else:
            raise ValueError("Both should be None, or both should be something.")

        X = np.array([self.X]).T
        Y = np.array(self.Y)

        self.gpr = GaussianProcessRegressor()
        self.gpr.fit(X, Y)

    def normal_acquisition(self, eta=0.1):
        # Implements the usual Prob. of Improvement
        mu, sigma = self.gpr.predict(
            self.domain.reshape(-1, 1),
            return_std=True
        )
        f_star = max(self.Y) # best performance seen so far.
        with np.errstate(divide="warn"):        
            prob_improvement = norm.cdf(
                (mu - f_star - eta) / sigma
            )

        return prob_improvement

    def expected_improvement(self, eta=0.01):
        mu, sigma = self.gpr.predict(
            self.domain.reshape(-1, 1),
            return_std=True
        )
        f_star = max(self.Y)
        print(f"mu {mu}")
        print(f"sigma {sigma}")
        with np.errstate(divide="warn"):
            numerator = (mu - f_star - eta)
            quot = numerator / sigma
            ei = numerator * norm.cdf(quot) + sigma * norm.pdf(quot)
            ei[sigma == 0] = 0

        return ei

    def custom_acquisition(self, eta=0.1):
        mu, sigma = self.gpr.predict(
            self.domain.reshape(-1, 1),
            return_std=True
        )
        f_star = max(self.Y)
        g_star = self._g(f_star) # best performance seen so far.

        print(sigma)
        with np.errstate(divide="warn"):
            prob_1 = norm.cdf(
                (mu - (self.goal - g_star) - eta) / sigma
            )
            prob_2 = norm.cdf(
                (mu - (self.goal - g_star) - eta) / sigma
            )
            prob_improvement = 1 - (prob_1 - prob_2)

        return prob_improvement

    def mean(self):
        mu, sigma = self.gpr.predict(
            self.domain.reshape(-1, 1),
            return_std=True
        )
        return mu

    def _g(self, u):
        return - np.abs(u - self.goal)

    def next(self):
        acquisition = self.expected_improvement()
        # acquisition = self.normal_acquisition()
        # acquisition = self.custom_acquisition()
        # print(np.max(acquisition))
        return self.domain[np.argmax(acquisition)]
