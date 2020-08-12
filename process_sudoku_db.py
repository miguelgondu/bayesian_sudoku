"""
Since loading the whole csv is very expensive,
this script dumps all the solved sudokus into a
json file in the same path as PATH_TO_SUDOKUS.
"""
import json
import pandas as pd

with open("./config.json") as fp:
    config = json.load(fp)

path_ = config['PATH_TO_SUDOKUS']

all_sudokus = pd.read_csv(
    f"{path_}/sudoku.csv"
    ).values[:, 1].astype(str).tolist()[:2000]

with open(f"{path_}/sudoku.json", "w") as fp:
    json.dump(all_sudokus, fp)
