"""
This script contains the utilities
for solution checking.
"""
import numpy as np
from math import sqrt

def parse_data(data):
    """
    Reconstructs the array from
    the string sent by the form.
    """
    data = {
        k: v for k, v in data.items() if "solved_sudoku" in k
    }

    if len(data) not in [4*4, 6*6, 9*9]:
        raise ValueError("Weird, we only handle 4x4, 6x6 and 9x9 sudokus.")

    size = int(sqrt(len(data)))
    solved_sudoku = []
    for i in range(size):
        solved_sudoku.append([])
        for j in range(size):
            if data[f"solved_sudoku[{i}][{j}]"] == "":
                data[f"solved_sudoku[{i}][{j}]"] = "0"

            solved_sudoku[i].append(
                int(data[f"solved_sudoku[{i}][{j}]"])
            )

    return solved_sudoku

def check_solution(board):
    """
    Parses the data and returns a
    bool that says whether the solution is a valid one or not.

    It receives the board as a numpy array.

    TODO: make this more verbose for the final version,
    and maybe add the table with red highlighting in when
    the solution checker outputs False.
    """
    # TODO: generic checks on whether all entries are numbers
    # and within range.
    size = len(board)

    # Checking rows and columns
    for i in range(size):
        row = board[i, :]
        column = board[:, i]

        if np.any((row > size).any() or (row < 1).any()):
            return False, f"Numbers out of range in row {i+1}!"
        
        row_set = set(row.tolist())
        if len(row_set) != size:
            return False, f"Numbers are repeating in row {i+1}!"
        
        if np.any((column > size).any() or (column < 1).any()):
            return False, f"Numbers out of range in column {i+1}!"

        column = set(column.tolist())
        if len(column) != size:
            return False, f"Numbers are repeating in column {i}!"
    
    rows_per_subblock = None
    columns_per_subblock = None

    subblock_rows = None
    subblock_columns = None

    if size == 4:
        subblock_rows = 2
        rows_per_subblock = 2
        subblock_columns = 2
        columns_per_subblock = 2
    elif size == 6:
        subblock_rows = 3
        rows_per_subblock = 2
        subblock_columns = 2
        columns_per_subblock = 3
    elif size == 9:
        subblock_rows = 3
        rows_per_subblock = 3
        subblock_columns = 3
        columns_per_subblock = 3
    else:
        raise ValueError(f"Got incorrect value for size: {size}. Was expecting 4, 6 or 9.")

    for i in range(subblock_rows):
        for j in range(subblock_columns):
            subblock = board[
                rows_per_subblock*i:rows_per_subblock*(i+1),
                columns_per_subblock*j:columns_per_subblock*(j+1),
            ]
            subblock = set(subblock.flatten().tolist())
            if len(subblock) != size:
                return False, f"Numbers are repeating in block {i+1}x{j+1}"

    return True, "The sudoku was solved correctly!"
