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

def g(y, goal):
    # Now assuming we're modeling y = log(t)
    # print(f"g: {- (np.exp(y) - goal) ** 2}")
    return - (np.exp(y) - goal) ** 2

class SudokuExperiment:
    """
    This class maintains every Sudoku Experiment.

    TODO:
    - Write this docstring.
    - Implement loading from a prior.
    """
    def __init__(self, size, goal, domain=None, name=None, curriculum=[], load_prior=True, debugging=False):
        self.timestamp = str(time.time()).replace(".", "")
        self.size = size
        self.goal = goal
        self.name = name
        if domain is not None:
            self.domain = domain
        else:
            if size == 4:
                self.domain = np.arange(4, 16) # TODO: check this out
            elif size == 6:
                self.domain = np.arange(8, 36)
            elif size == 9:
                self.domain = np.arange(17, 81)

        # TODO: this is a hot fix for not loading the prior
        # I should implement this by just passing the path to the prior
        # or by passing a float (e.g. when we want a flat
        # prior at 0 or something).
        print("Loading prior.")
        self.prior_array = np.loadtxt(
            f"{PATH_TO_PRIORS}/{size}x{size}.csv", delimiter=","
        )
        if load_prior:
            self.prior = {
                int(self.prior_array[k, 0]): float(self.prior_array[k, 1]) for k in range(len(self.prior_array))
            }
        else:
            self.prior = {
                int(self.prior_array[k, 0]): 0 for k in range(len(self.prior_array))
            }
        
        print("Creating the sudoku corpus.")
        self.sudoku_corpus = self._create_sudoku_corpus(
            f"{PATH_TO_SUDOKUS}/sudokus_{size}x{size}.csv"
        )

        print("Instantiating the GPR.")
        self.kernel = 1*RBF(length_scale=1) + WhiteKernel(noise_level=np.log(2))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)

        self.curriculum = curriculum

        self.next_hints = None
        self.next_sudoku_ = None

        self.times = []
        self.log_times = []
        self.hints = []
        self.sudokus = []

        self.real_map = None
        self.variance_map = None

        self.debugging = debugging

    def to_json(self):
        """
        Drops everything into a JSON object.
        """
        if self.real_map is None:
            real_map = None
            variance_map = None
        else:
            real_map = {k: float(v) for k, v in self.real_map.items()}
            variance_map = {k: float(v) for k, v in self.variance_map.items()}

        as_json = {
            "name": self.name,
            "size": self.size,
            "goal": self.goal,
            "domain": self.domain.astype(int).tolist(),
            "prior": {int(k): float(v) for k, v in self.prior.items()},
            "prior_g": [float(self._g(t)) for t in self.prior.values()],
            "times": [float(t) for t in self.times],
            "log_times": [float(t) for t in self.log_times],
            "hints": [int(h) for h in self.hints],
            "sudokus": self.sudokus,
            "real_map": real_map,
            "variance_map": variance_map,
            "next_hints": self.next_hints,
            "next_sudoku_": self.next_sudoku_
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
            filename = f"{self.timestamp}_sudoku_experiment_{self.name}"

        with open(f"{PATH_TO_EXPERIMENTS}/{filename}.json", "w") as fp:
            json.dump(as_json, fp)
    
    @classmethod
    def from_json(cls, json_obj):
        """
        Reconstructs the object from a json-like object
        (assuming it's similar to the one dumped by self.to_json())

        This is not optimized.
        """
        print("Reconstructing from:")
        print(json_obj)

        size = json_obj["size"]
        goal = json_obj["goal"]
        name = json_obj["name"]
        domain = np.array(json_obj["domain"])

        se = cls(size, goal, domain=domain, name=name, load_prior=False)

        se.hints = json_obj["hints"]
        se.times = json_obj["times"]
        se.log_times = json_obj["log_times"]
        se.sudokus = json_obj["sudokus"]

        se.next_hints = json_obj["next_hints"]
        se.next_sudoku_ = json_obj["next_sudoku_"]

        se.prior = {
            int(k): float(v) for k, v in json_obj["prior"].items()
        }
        real_map = None
        variance_map = None

        if json_obj["real_map"] is not None:
            real_map = {
                int(k): float(v) for k, v in json_obj["real_map"].items()
            }
            variance_map = {
                int(k): float(v) for k, v in json_obj["variance_map"].items()
            }
        
        se.real_map = real_map
        se.variance_map = variance_map

        print("Reconstructed version:")
        print(f"{se.to_json()}")

        return se

    def _create_sudoku_corpus(self, path):
        """
        We use the csv that can be downloaded from:
        https://www.kaggle.com/bryanpark/sudoku
        """
        # The way the Kaggle's database of sudokus is
        # structured is by having two columns: puzzle,solved.
        # we grab the solved ones.
        with open(f"{PATH_TO_SUDOKUS}/sudoku.json") as fp:
            corpus = json.load(fp)

        random.shuffle(corpus)
        # Transform them to sudokus
        corpus = [
            string_to_sudoku(s) for s in corpus
        ]

        return corpus
    
    def fit_gpr(self):
        X = np.array([self.hints.copy()]).T
        # The GP is modeling the difference g(t(x)) - g(prior(x))
        # which should be close to 0.
        # No. The GP will model the difference t(x) - prior(x). The
        # bending will occur at acquisition time.
        Y = np.array([
            log_time - self.prior[hint] for log_time, hint in zip(self.log_times, self.hints)
        ])

        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
        if len(X) > 0:
            print(f"Fitting the GPR with")
            print(f"X: {X}, Y: {Y}")
            self.gpr.fit(X, Y)

    def next_sudoku(self):
        """
        Uses the inner Bayesian Optimization to serve the next sudoku.
        If curriculum has any numbers, it will serve them in order until
        they have been depleted.
        """

        """
        There should be a better way of implementing this, something like
        "deploy" in the ITAE case.
        """

        # Deploy one from the curriculum, if there are any still there.
        if len(self.curriculum) > 0:
            next_hints = self.curriculum.pop(0)
        else:
            next_hints = self.acquisition()

        self.next_hints = int(next_hints)
        next_sudoku = self.create_sudoku(next_hints)
        self.next_sudoku_ = sudoku_to_string(np.array(next_sudoku))

        return next_sudoku
    
    def acquisition(self):
        '''
        Returns the hint that maximizes the expected improvement.
        
        This expected improvement takes into account the actual
        objective function, which is a bent version of the time
        we are modeling using self._g = - (t - goal) ** 2.

        TODO: This docstring will change once we model log(t)
        instead of t.
        '''
        if self.debugging:
            return 80
        ei = self.expected_improvement()
        return self.domain[np.argmax(ei)]

    def expected_improvement(self, return_mu_and_sigma=False):
        """
        Computes the EI using self.domain.
        """
        n_samples = 10000

        if len(self.times) > 0:
            max_so_far = max([self._g(log_t) for log_t in self.log_times])
        else:
            # TODO: Think if I should be doing something else here.
            max_so_far = -9999

        print("self.prior")
        print(self.prior)
        prior = np.array([self.prior[h] for h in self.domain]).reshape(-1, 1)

        # Assure the GP has been trained properly:
        self.fit_gpr()
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
        return g(t, self.goal)

    def register_time(self, time_it_took, save=True):
        print(f"Registering time {time_it_took} for {self.next_hints} hints.")
        print(f"Original prior said it should take {np.exp(self.prior[self.next_hints])}.")
        self.hints.append(self.next_hints)
        self.times.append(time_it_took)
        self.log_times.append(np.log(time_it_took))
        self.sudokus.append(self.next_sudoku_)

        self.fit_gpr()
        self.update_real_and_variance_maps()

        prediction = np.exp(self.real_map[self.next_hints])
        print(f"Our new prediction for {self.next_hints} is {prediction}.")
        if save:
            self.save()

    def update_real_and_variance_maps(self):
        # Recompute the new mean_real and variance
        # using current GP regression.
        hints = list(self.prior.keys())
        prior_times = np.array([self.prior[h] for h in hints])

        mu, sigma = self.gpr.predict(
            np.array([hints]).T,
            return_std=True
        )

        real_values = mu + prior_times
        variance_values = sigma.T

        # print(f"mu (log): {mu}")
        # print(f"real values (log): {real_values}")
        # print(f"variance: {variance_values}")

        real_map = {}
        variance_map = {}
        for i, hint in enumerate(hints):
            real_map[hint] = real_values[i]
            variance_map[hint] = variance_values[i]
        
        self.real_map = real_map
        self.variance_map = variance_map

    def create_sudoku(self, hints):
        # Get a sudoku we haven't seen in the past (TODO: optimize)
        a_sudoku = random.choice([
            s for s in self.sudoku_corpus if s not in self.sudokus
        ])
        a_sudoku = a_sudoku.copy()

        # Assuming the sudoku is solved
        positions = list(product(range(self.size), repeat=2))
        random.shuffle(positions)

        for k in range(self.size ** 2 - hints):
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
        # I'll implement a gpr on self.real_map
        to_plot = None
        if self.real_map is not None:
            title1 = "real map"
            to_plot = self.real_map
            sigma = [self.variance_map[h] for h in self.domain]
        else:
            title1 = "prior"
            to_plot = self.prior
            sigma = [0 for h in self.domain]
        
        values = [
            to_plot[h] for h in self.domain
        ]
        values = np.array(values)
        sigma = np.array(sigma)

        if self.real_map is not None:
            ax1.plot(self.hints, self.times, "ok", label="Observations")
            ax1.plot(self.domain, np.exp(values), 'b-', label='GP')
            ax1.fill_between(self.domain, np.exp(values-sigma).squeeze(), np.exp(values+sigma).squeeze(), alpha=0.5)
        else:
            ax1.plot(self.domain, np.exp(values), 'b-', label='GP')
        
        ax1.axhline(y=self.goal)
        ax1.set_title(title1)
        ax1.set_xlabel("Hints")
        ax1.set_ylabel("Predicted time [sec]")


        prior = np.array([self.prior[h] for h in self.domain]).reshape(-1, 1)
        # ax2 for acquisition function
        ei, mu, sigma, g_samples = self.expected_improvement(return_mu_and_sigma=True)
        # print("visualize:")
        # print(f"mu: {mu}")
        # print(f"sigma: {sigma}")
        # print(f"g_samples: {g_samples.shape}")
        if self.real_map is not None:
            ax2.plot(self.domain, ei, "rx")
            ax2.set_title("Acq. function (EI).")
            ax2.set_xlabel("Hints")

            ax3.plot(self.domain, mu, "gx")
            ax3.set_title("Real time - prior")
            ax3.set_xlabel("Hints")

            ax4.plot(self.domain, -self._g(np.array(values)), 'b-')
            ax4.plot(self.domain, -g_samples, 'b.', alpha=0.002)
            ax4.set_yscale("log")
            ax4.set_title("- obj. function and samples")
            ax4.set_xlabel("Hints")

        plt.tight_layout()

        if self.name is None:
            plt.savefig(f"{PATH_TO_IMAGES}/gp_at_sudoku_{len(self.sudoku)}.jpg")
        else:
            plt.savefig(f"{PATH_TO_IMAGES}/gp_{self.name}_at_sudoku_{len(self.sudokus)}.jpg")

        plt.close()
