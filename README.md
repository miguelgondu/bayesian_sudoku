# Sudoku web-app with Bayesian Optimization

This repo contains the code for the first experiment of the paper we plan to write for AAAI 2020. This experiment, summarized, consists of a sudoku web app that optimizes the sudokus it serves for them to be solved in a given time.

## Running the sudoku web app

Create a virtual environment (if you want) and install the requirements with

```
pip install -r requirements.txt
```

Run

```
python app.py
```

and load the webpage that's being served. That should work!

## After running the sudoku web app

After running some iterations, you can find visualizations of the time-curve getting updated in `./data/images`.

TODO: add some details about the database.