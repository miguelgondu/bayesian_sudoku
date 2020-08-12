import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from flask import Flask, render_template, url_for, request

from solution_checking import parse_data, check_solution
from sudoku_experiment import SudokuExperiment

app = Flask(__name__)

# Squelch a warning
plt.switch_backend('Agg')

print("Creating the Sudoku Experiment object")
size = 9
goal = 2 * 60
kappa = 0.5
se = SudokuExperiment(
    size,
    goal,
    name="local_testing"
)

start = None
counter = 0

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/next", methods=["POST"])
def next():
    # TODO: fix these globals when the database has been
    # implemented.
    global start
    global counter
    global se

    data = request.form
    username = data["username"]
    if username not in se.name:
        se.name = username + "_" + se.name
    start = time.time()

    # TODO: add logic of registering the username and
    # querying for the next sudoku here.
    # It should make it so that no sudoku is presented
    # more than once.

    next_sudoku = se.next_sudoku()
    # print(se.acquisition())
    print(f"next hints: {se.next_hints}")
    print(f"next sudoku: {next_sudoku}")
    return render_template("next.html", sudoku=next_sudoku, username=username)


@app.route("/solution", methods=["POST"])
def solution():
    """
    This view shows an alert of wether the sudoku 
    was solved correctly or not. Maybe I should add a
    button that POSTs to a /next page.
    """

    # TODO: fix this after implementing the database
    # logic.
    global start
    global se

    # Get the time it took to solve the puzzle
    final = time.time()
    try:
        time_it_took = final - start
        print(f"it took {time_it_took}")
    except TypeError:
        print("Couldn't track time. :(")
    
    start = None

    # TODO: store the sudoku and the time the player
    # took to solve it in the database.

    data = request.form
    board = np.array(parse_data(data))
    solved, message = check_solution(board)

    # I need to POST the amount of numbers for now,
    # or define it as a global variable while I
    # implement the database logic.
    if solved:
        # Fitting the GP
        print("Registering time.")
        se.register_time(time_it_took)

        # Plotting the mean of the GP and the acquisition for debugging
        print("Visualizing.")
        se.visualize()
    else:
        # TODO: remove the last hint, or re-plan how the hints are
        # being kept.
        # Implement what happens when the solution wasn't good.
        # What's that exactly?
        pass

    # TODO: implement what happens if not solved.
    return render_template("solution.html", solved=solved, message=message, username=data["username"])


if __name__ == "__main__":
    print("Serving the web app")
    app.run(debug=True)
