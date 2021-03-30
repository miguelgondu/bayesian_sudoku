from typing import List
import json
import random
from itertools import product

import numpy as np
from sklearn.linear_model import LinearRegression

from experiment import Experiment
from sudoku_utilities import string_to_sudoku

PATH_TO_SUDOKUS = '.'


class LinearRegressionExperiment(Experiment):
    def __init__(self, goal: float, hints: List[float] = [], times: List[float] = [], curriculum: List[int] = []):
        self.goal = goal
        self.hints = hints
        self.times = times
        self.curriculum = []

        self.domain = np.arange(17, 81)
        self.prior = np.array([self.prior_fn(h) for h in self.domain])

        print("Creating the sudoku corpus.")
        self.sudoku_corpus = self._create_sudoku_corpus()
    
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

    def prior_fn(self, hint):
        return np.log(600 + ((600 - 3) / (17 - 80)) * (hint - 17))

    def next_sudoku(self):
        if len(self.curriculum) > 0:
            next_hints = self.curriculum.pop(0)
        elif len(self.hints) == len(self.times) + 1:
            # We're still testing certain hint,
            # we haven't recorded time for it
            # (e.g. refreshing the /next page).
            next_hints = self.hints[-1]
        else:
            next_hints = self._acquisition()
        
        return self._create_sudoku(next_hints)

    def _fit_linear_reg(self):
        # Returs actual time.
        hints = np.array(self.hints).reshape(-1, 1)
        if min(len(self.hints), len(self.times)) > 0:
            y_diff = [
                np.log(time) - self.prior_fn(hint) for (hint, time) in zip(hints, self.times)
            ]
            reg = LinearRegression().fit(hints, y_diff)

            coef = reg.coef_.copy()
            intercept = reg.intercept_
        else:
            coef = 0
            intercept = 0
        
        t = np.exp([(coef*x + intercept) + self.prior_fn(x) for x in self.domain])
        return t.flatten()

    def _acquisition(self):
        # Returns the x \in domain s.t. the linear regression
        # is closest to goal.
        # Returns the hint \in domain
        # that's closest to goal in time.
        t = self._fit_linear_reg()
        return self.domain[np.argmin(np.abs(t - self.goal))]

    def _create_sudoku(self, hints):
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
