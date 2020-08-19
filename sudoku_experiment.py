"""
This script maintains the SudokuExperiment object,
an abstraction of the process of serving sudokus and
optimizing for a particular time using a Gaussian Process
Regression.
"""
import json
import random
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sudoku_utilities import string_to_sudoku

PATH_TO_SUDOKUS = '.'

class SudokuExperiment:
    """
    SudokuExperiment maintains the objective time at goal, and keeps
    track of the sudokus that have been played (according to an encoding)
    and the time it took for the player to solve them. It uses this information
    to train and maintain a Gaussian Process Regression, which is then used
    for Bayesian Optimization.

    Bayesian Optimization relies on a probabilistic model (GPR in our case) and
    an acquisition function. Since we're not trying to optimize time (i.e. we're
    not trying to find the max time, making sudokus more and more difficult),
    we "bend" the predicted time with g(t) = -(t - goal)**2. This function is now
    optimized at {goal}.

    Another detail: we model log(time) instead of time to avoid negative times.
    """

    def __init__(self, goal, hints=[], times=[], name="adaptive", curriculum=[], debugging=False):
        self.goal = goal
        self.name = name

        # From now on, we'll assume that sudokus are 9x9.
        self.domain = np.arange(17, 81)

        self.prior = np.array([self.prior_fn(h) for h in self.domain])

        print("Creating the sudoku corpus.")
        self.sudoku_corpus = self._create_sudoku_corpus()

        self.curriculum = curriculum

        # hints, times and log_times are always lists.
        self.hints = hints
        self.times = times
        self.log_times = [np.log(t) for t in self.times]

        # Since EI is stochastic, we need to maintain this
        # for the communication between sites:

        # These will be arrays of len(self.domain)
        # containing our current approximations.
        # self.real_map = None
        # self.variance_map = None

        print("Instantiating the GPR.")
        self.kernel = 1 * RBF(length_scale=1) + WhiteKernel(noise_level=np.log(2))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)

        if len(self.hints) == len(self.times) and len(self.times) > 0:
            self._fit_gpr()

        self.debugging = debugging

    def prior_fn(self, hint):
        return np.log(600 + ((600 - 3) / (17 - 80)) * (hint - 17))

    def _create_sudoku_corpus(self):
        # Load the 2000 sudokus
        with open(f"{PATH_TO_SUDOKUS}/sudoku.json") as fp:
            corpus = json.load(fp)

        # Shuffle them
        random.shuffle(corpus)

        # Transform them to sudokus
        corpus = [
            string_to_sudoku(s) for s in corpus
        ]

        return corpus

    def _fit_gpr(self):
        """
        Fits a Gaussian Process on log(t(x)) - prior(x),
        where x is the amount of digits we're giving
        the player.
        """

        X = np.array([self.hints.copy()]).T
        Y = np.array([
            log_time - self.prior[np.where(self.domain == hint)] for log_time, hint in zip(self.log_times, self.hints)
        ])

        if len(X) > 0 and len(Y) > 0:
            print(f"Fitting the GPR with")
            print(f"X: {X}, Y: {Y}")
            self.gpr.fit(X, Y)

    def next_sudoku(self):
        """
        Uses the inner Bayesian Optimization to serve the next sudoku.
        If curriculum has any numbers, it will serve them in order until
        they have been depleted.
        """
        if len(self.curriculum) > 0:
            next_hints = self.curriculum.pop(0)
        elif len(self.hints) == len(self.times) + 1:
            # We're still testing certain hint,
            # we haven't recorded time for it
            # (e.g. refreshing the /next page).
            next_hints = self.hints[-1]
        else:
            next_hints = self.acquisition()

        print(f"Deploying a sudoku with {next_hints} hints.")
        next_sudoku = self.create_sudoku(next_hints)

        return next_sudoku

    def acquisition(self):
        '''
        This function trains the GPR and returns the hint
        that maximizes the expected improvement.
        
        This expected improvement takes into account the actual
        objective function, which is a bent version of the time
        we are modeling using self._g = - (t - goal) ** 2.
        '''
        if self.debugging:
            return 80

        ei = self.expected_improvement()
        next_hints = int(self.domain[np.argmax(ei)])

        self.hints.append(next_hints)
        return next_hints

    def expected_improvement(self, return_mu_and_sigma=False):
        """
        Computes the EI by sampling.
        """
        n_samples = 10000

        if len(self.times) > 0:
            max_so_far = max([self._g(log_t) for log_t in self.log_times])
        else:
            max_so_far = -9999

        prior = self.prior.reshape(-1, 1)

        mu, sigma = self.gpr.predict(self.domain.reshape(-1, 1), return_std=True)
        mu_gp = mu.copy()
        sigma = sigma.reshape(-1, 1)
        mu = prior + mu.reshape(-1, 1)

        g_samples = self._g(mu + sigma * np.random.randn(mu.shape[0], n_samples))
        ei = np.mean(np.maximum(0, g_samples - max_so_far), axis=1)

        if return_mu_and_sigma:
            return (ei, mu_gp, sigma, g_samples)
        else:
            return ei

    def _g(self, t):
        return - (np.exp(t) - self.goal) ** 2

    def compute_real_and_variance_maps(self):
        """
        Could be optimized further.
        """

        mu, sigma = self.gpr.predict(
            self.domain.reshape(-1, 1),
            return_std=True
        )
        sigma = sigma.reshape(-1, 1)

        real_map = mu + self.prior.reshape(-1, 1)
        variance_map = sigma

        return real_map, variance_map

    def create_sudoku(self, hints):
        # TODO: once we implement a database, we can change this
        # s.t. we don't serve the same sudoku twice for a player.
        # Chances are slim, though.

        a_sudoku = random.choice(self.sudoku_corpus)
        a_sudoku = a_sudoku.copy()

        # Assuming the sudoku is solved
        positions = list(product(range(9), repeat=2))
        random.shuffle(positions)

        for k in range(9 ** 2 - hints):
            i, j = positions[k]
            a_sudoku[i][j] = 0

        return a_sudoku

    def visualize(self, domain=None):
        if self.debugging:
            return

        _, axes = plt.subplots(2, 2, figsize=(10, 10))

        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]

        # ax1 for mean
        to_plot = None
        if len(self.times) > 0:
            real_map, variance_map = self.compute_real_and_variance_maps()
            title1 = "real map"
            to_plot = real_map
            sigma = variance_map
        else:
            title1 = "prior"
            to_plot = self.prior
            sigma = [0 for h in self.domain]

        if title1 != "prior":
            ax1.plot(self.hints, self.times, "ok", label="Observations")
            ax1.plot(self.domain, np.exp(to_plot), 'b-', label='GP')
            print("np.exp(to_plot-sigma).squeeze()")
            print(f"{np.exp(to_plot - sigma).squeeze()}")
            print(f"to_plot")
            print(f"{to_plot}")
            print(f"sigma")
            print(f"{sigma}")
            ax1.fill_between(self.domain, np.exp(to_plot - sigma).squeeze(), np.exp(to_plot + sigma).squeeze(), alpha=0.5)
        else:
            ax1.plot(self.domain, np.exp(to_plot), 'b-', label='GP')

        ax1.axhline(y=self.goal)
        ax1.set_title(title1)
        ax1.set_xlabel("Hints")
        ax1.set_ylabel("Predicted time [sec]")

        prior = self.prior.reshape(-1, 1)
        # ax2 for acquisition function
        ei, mu, sigma, g_samples = self.expected_improvement(return_mu_and_sigma=True)
        # print("visualize:")
        # print(f"mu: {mu}")
        # print(f"sigma: {sigma}")
        # print(f"g_samples: {g_samples.shape}")
        if title1 != "prior":
            ax2.plot(self.domain, ei, "rx")
            ax2.set_title("Acq. function (EI).")
            ax2.set_xlabel("Hints")

            ax3.plot(self.domain, mu, "gx")
            ax3.set_title("Real time - prior")
            ax3.set_xlabel("Hints")

            ax4.plot(self.domain, -self._g(np.array(to_plot)), 'b-')
            ax4.plot(self.domain, -g_samples, 'b.', alpha=0.002)
            ax4.set_yscale("log")
            ax4.set_title("- obj. function and samples")
            ax4.set_xlabel("Hints")

        plt.tight_layout()

        if self.name is None:
            plt.savefig(f"gp_at_sudoku_{len(self.times)}.jpg")
        else:
            plt.savefig(f"gp_{self.name}_at_sudoku_{len(self.times)}.jpg")

        plt.close()
