import numpy as np
from math import sqrt, ceil, floor

def sudoku_to_string(sudoku):
    if not isinstance(sudoku, np.ndarray):
        sudoku = np.array(sudoku)
    sudoku = sudoku.astype(str).flatten().tolist()
    str_sudoku = "".join(sudoku)
    return str_sudoku

def string_to_sudoku(s):
    size = sqrt(len(s))

    # Check if the size is an integer
    assert floor(size) == ceil(size), "length of the string should be a square."
    size = int(size)

    rows = [
        s[k*size:(k+1)*size] for k in range(size)
    ]

    sudoku = [
        [int(c) for c in row] for row in rows
    ]
    return sudoku
