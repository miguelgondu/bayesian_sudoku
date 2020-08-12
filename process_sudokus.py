import pandas as pd
import json

from sudoku_solver import solve
from sudoku_experiment import PATH_TO_SUDOKUS
from sudoku_utilities import sudoku_to_string

# Processing the 4x4 and 6x6

s_4x4 = pd.read_csv(
    f"{PATH_TO_SUDOKUS}/sudokus_4x4.csv",
    delimiter=";",
    header=None).fillna(0).values.astype(int)

s_6x6 = pd.read_csv(
    f"{PATH_TO_SUDOKUS}/sudokus_6x6.csv",
    delimiter=";",
    header=None).fillna(0).values.astype(int)

all_sudokus_4x4 = [
    s_4x4[4*k:4*(k+1)] for k in range(len(s_4x4) // 4)
]

all_sudokus_6x6 = [
    s_6x6[6*k:6*(k+1)] for k in range(len(s_6x6) // 6)
]

for sudoku in all_sudokus_4x4:
    solved = solve(sudoku, size=4)
    assert solved

for sudoku in all_sudokus_6x6:
    solved = solve(sudoku, size=6)
    assert solved

processed_sudokus_4x4 = [
    sudoku_to_string(s) for s in all_sudokus_4x4
]

processed_sudokus_6x6 = [
    sudoku_to_string(s) for s in all_sudokus_6x6
]

# Processing the 9x9

s_9x9 = pd.read_csv(f"{PATH_TO_SUDOKUS}/sudokus_9x9.csv")

processed_sudokus_9x9 = s_9x9.values[:500, 1].tolist()

print(processed_sudokus_9x9)

with open(f"{PATH_TO_SUDOKUS}/sudokus_4x4.json", "w") as fp:
    json.dump(processed_sudokus_4x4, fp)

with open(f"{PATH_TO_SUDOKUS}/sudokus_6x6.json", "w") as fp:
    json.dump(processed_sudokus_6x6, fp)

with open(f"{PATH_TO_SUDOKUS}/sudokus_9x9.json", "w") as fp:
    json.dump(processed_sudokus_9x9, fp)
