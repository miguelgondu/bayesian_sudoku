import json
import random
from itertools import product

from sudoku_solver import solve
from sudoku_utilities import string_to_sudoku
import numpy as np


def _create_sudoku_corpus():
    # Load the 2000 sudokus
    with open(f"./sudoku.json") as fp:
        corpus = json.load(fp)

    # Shuffle them
    random.shuffle(corpus)

    # Transform them to sudokus
    corpus = [string_to_sudoku(s) for s in corpus]

    return corpus


corpus = _create_sudoku_corpus()


a_sudoku = random.choice(corpus)
a_sudoku = a_sudoku.copy()


def create_sudokus_for_gif(a_sudoku, hints):
    # TODO: once we implement a database, we can change this
    # s.t. we don't serve the same sudoku twice for a player.
    # Chances are slim, though.

    # Assuming the sudoku is solved
    positions = list(product(range(9), repeat=2))
    random.shuffle(positions)

    for k in range(9 ** 2 - hints):
        i, j = positions[k]
        a_sudoku[i][j] = 0
        if k % 5 == 0 or k == 9 ** 2 - hints - 1:
            sudoku_as_array = np.array(a_sudoku)
            np.savetxt(
                f"sudoku_hints_{hints}_{81 - k - 1}.csv",
                sudoku_as_array,
                delimiter=",",
                fmt="%1d",
            )

    return a_sudoku


create_sudokus_for_gif(a_sudoku, 56)
