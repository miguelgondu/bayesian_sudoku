"""
This script takes the Kaggle competition ones,
solves them, and randomly removes numbers, verifying
solvability.
"""
from math import sqrt
from itertools import product
import random

import pandas as pd
import numpy as np

from sudoku_solver import solve

def parse_sudoku(sudoku_string):
    """
    This function takes a string from the Kaggle
    database and parses it as an array.

    TODO: verify that size is within expectations (4, 6, 9).
    """
    size = int(sqrt(len(sudoku_string)))

    sudoku = []
    for j in range(size):
        row = []
        for i in range(size):
            row.append(sudoku_string[size*j + i])
        sudoku.append(row)
    
    return sudoku

def prepare_solved_sudoku(sudoku, free_spots):
    """
    This function takes a solved sudoku
    and removes numbers until getting enough
    free spots (according to free_spots).

    It also verifies that the resulting sudoku
    is solvable before returning it.

    It assumes that sudoku is array-like.
    """
    og_sudoku = sudoku.copy()

    # TODO: fix this for 6x6 and 4x4
    if free_spots > 81 - 17:
        raise ValueError("Leave at least 17 hints!")

    size = len(sudoku)
    all_positions = list(product(range(size), range(size)))
    random.shuffle(all_positions)

    # print(all_positions)
    for k in range(free_spots):
        i, j = all_positions[k]
        sudoku[i][j] = "0"

    copy = sudoku.copy()
    solve(copy) # Solve it using the solver

    # and check if it's equal to the original solution
    assert og_sudoku == copy

    # cast to int
    sudoku = np.array(sudoku, dtype=int).tolist()

    return sudoku

# if __name__ == "__main__":
#     sudokus = pd.read_csv(
#         "/Users/migd/Projects/first_adaptive_game_data/sudokus/sudokus_kaggle.csv"
#     )
#     sudokus = sudokus.sample(frac=1).reset_index()

#     print(sudokus.head())

#     sudoku = sudokus.loc[0, "solutions"]
#     print(sudoku)

#     sudoku = parse_sudoku(sudoku)
#     print(np.array(sudoku))

#     sudoku = prepare_solved_sudoku(sudoku, 20)
#     print(np.array(sudoku))
