import json
import random
from itertools import product

import numpy as np

from experiment import Experiment
from sudoku_utilities import string_to_sudoku

PATH_TO_SUDOKUS = '.'


class BinarySearchExperiment(Experiment):
    """
    Binary search for a sudoku w. the right amount of hints
    """

    def __init__(self, goal, hints=[], times=[], name="binary", curriculum=[]):
        self.goal = goal
        self.name = name

        # From now on, we'll assume that sudokus are 9x9.
        self.domain = np.arange(17, 81)

        print("Creating the sudoku corpus.")
        self.sudoku_corpus = self._create_sudoku_corpus()

        self.curriculum = curriculum

        # hints, times and log_times are always lists.
        self.hints = hints
        self.times = times

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

    def next_sudoku(self):
        if len(self.curriculum) > 0:
            next_hints = self.curriculum.pop(0)
        else:
            upper = 81
            lower = 17
            for t in self.times:
                midpoint = lower + int(round((upper - lower) / 2))
                if t > self.goal:
                    lower = midpoint
                else:
                    upper = midpoint

            next_hints = lower + int(round((upper - lower) / 2))

        print(f"Deploying a sudoku with {next_hints} hints.")
        next_sudoku = self.create_sudoku(next_hints)

        return next_sudoku

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
