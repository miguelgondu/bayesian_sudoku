"""
This script maintains the SudokuExperiment object,
an abstraction of the process of serving sudokus and
optimizing for a particular time using a Gaussian Process
Regression.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import time
from itertools import product

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm

from sudoku_utilities import string_to_sudoku, sudoku_to_string
from sudoku_solver import solve

with open("config.json") as fp:
    config = json.load(fp)

PATH_TO_PRIORS = config["PATH_TO_PRIORS"]
PATH_TO_SUDOKUS = config["PATH_TO_SUDOKUS"]
PATH_TO_EXPERIMENTS = config["PATH_TO_EXPERIMENTS"]
PATH_TO_IMAGES = config["PATH_TO_IMAGES"]

class SudokuExperiment:
    """
    This class maintains every Sudoku Experiment.

    TODO:
    - Write this docstring.
    - Implement loading from a prior.
    """
    def __init__(self, goal, hints=[], times=[], name="adaptive", curriculum=[], prior_path=None, debugging=False):
        self.goal = goal
        self.name = name
        
        # From now on, we'll assume that sudokus are 9x9.
        self.domain = np.arange(17, 81)

        print("Loading prior.")
        if prior_path is None:
            prior_path = f"{PATH_TO_PRIORS}/9x9.csv"
        self.prior = np.loadtxt(prior_path)
        
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
        self.kernel = 1*RBF(length_scale=1) + WhiteKernel(noise_level=np.log(2))
        
        if len(self.hints) == len(self.times) and len(self.times) > 0:
            self.create_and_fit_gpr()
            # self.update_real_and_variance_maps()

        self.debugging = debugging

    def to_json(self):
        """
        Drops everything into a JSON object.
        """
        as_json = {
            "name": self.name,
            "goal": self.goal,
            "hints": [int(h) for h in self.hints],
            "times": [float(t) for t in self.times]
        }
        print("Trying to store this:")
        print(as_json)
        return as_json

    def save(self, filename=None):
        """
        Drops everything into a JSON object and saves it.

        filename: without ".json" at the end.
        """
        as_json = self.to_json()

        if filename is None:
            filename = f"{self.name}_sudoku_experiment"

        with open(f"{PATH_TO_EXPERIMENTS}/{filename}.json", "w") as fp:
            json.dump(as_json, fp)

    @classmethod
    def from_json(cls, json_obj):
        """
        Reconstructs the object from a json-like object
        (assuming it's similar to the one dumped by self.to_json())
        """
        print("Reconstructing from:")
        print(json_obj)

        goal = json_obj["goal"]
        name = json_obj["name"]
        hints = json_obj["hints"]
        times = json_obj["times"]

        se = cls(goal, hints=hints, times=times, name=name)
        return se

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
    
    def create_and_fit_gpr(self):
        """
        Fits a Gaussian Process on log(t(x)) - prior(x),
        where x is the amount of digits we're giving
        the player.
        """

        X = np.array([self.hints.copy()]).T
        Y = np.array([
            log_time - self.prior[np.where(self.domain == hint)] for log_time, hint in zip(self.log_times, self.hints)
        ])

        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
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

        # Assure the GP has been trained properly:
        self.create_and_fit_gpr()
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

    def register_time(self, time_it_took):
        """
        This function registers time in self.time (and log
        time accordingly). It should only be called if the
        sudoku was properly solved.
        """
        print(f"Registering time {time_it_took} for {self.hints[-1]} hints.")
        self.times.append(time_it_took)
        self.log_times.append(np.log(time_it_took))

    # def update_real_and_variance_maps(self):
    #     # Recompute the new mean_real and variance
    #     # using current GP regression.
    #     hints = self.domain.tolist()
    #     prior = np.array([self.prior[h] for h in hints])

    #     mu, sigma = self.gpr.predict(
    #         np.array([hints]).T,
    #         return_std=True
    #     )

    #     real_values = mu + prior
    #     variance_values = sigma.T

    #     # print(f"mu (log): {mu}")
    #     # print(f"real values (log): {real_values}")
    #     # print(f"variance: {variance_values}")

    #     real_map = {}
    #     variance_map = {}
    #     for i, hint in enumerate(hints):
    #         real_map[hint] = real_values[i]
    #         variance_map[hint] = variance_values[i]
        
    #     self.real_map = real_map
    #     self.variance_map = variance_map

    def compute_real_and_variance_maps(self):
        """
        Could be optimized further.
        """
        self.create_and_fit_gpr()

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
            print(f"{np.exp(to_plot-sigma).squeeze()}")
            print(f"to_plot")
            print(f"{to_plot}")
            print(f"sigma")
            print(f"{sigma}")
            ax1.fill_between(self.domain, np.exp(to_plot-sigma).squeeze(), np.exp(to_plot+sigma).squeeze(), alpha=0.5)
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
            plt.savefig(f"{PATH_TO_IMAGES}/gp_at_sudoku_{len(self.times)}.jpg")
        else:
            plt.savefig(f"{PATH_TO_IMAGES}/gp_{self.name}_at_sudoku_{len(self.times)}.jpg")

        plt.close()
