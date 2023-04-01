# Sudoku web-app with Bayesian Optimization

This repository contains the code for the web application used in the first experiment of the paper [*Fast Game Content Adaptation Through Bayesian-based Player Modelling*](https://arxiv.org/abs/2105.08484), published in CoG 2021. This experiment consists of a sudoku web app that optimizes the sudokus it serves for them to be solved in a given time.

## A guide to the code

The experiments are implemented in `binary_search.py`, `linear_regression_experiment.py` and `sudoku_experiment.py`. These contain the logic behind serving a sudoku given a player's playtrace.

The app itself is built using Flask. The relevant bits are in `app.py`, `templates/` and `trials.py` for the database connections (implemented with PostgreSQL).
