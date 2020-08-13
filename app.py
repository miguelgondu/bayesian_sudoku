import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from flask import Flask, render_template, url_for, request, session

from solution_checking import parse_data, check_solution
from sudoku_experiment import SudokuExperiment

app = Flask(__name__)

# Squelch a warning
plt.switch_backend('Agg')

# This will need to get changed to os.environ
# when deploying with Heroku.
with open("config.json") as fp:
    config = json.load(fp)
    secret_key = config["SECRET_KEY"]

app.secret_key = secret_key

@app.route("/")
def root():
    exp_id = str(time.time()).replace(".", "")
    session["experiment_id"] = exp_id
    print(f"Got experiment id: {session['experiment_id']}")

    print("Creating the Sudoku Experiment object")
    goal = 2 * 60
    se = SudokuExperiment(
        goal,
        name=f"{exp_id}"
        # debugging=True
    )
    print("Storing it in the session")
    session["se"] = se.to_json()

    print(session["se"])
    session["start"] = None

    return render_template("index.html")

@app.route("/next")
def next():
    session["start"] = time.time()
    session["final"] = None
    se = SudokuExperiment.from_json(session["se"])

    # This operation stores one hint in self.hints.
    next_sudoku = se.next_sudoku()
    session["se"] = se.to_json()

    print(f"next sudoku: {next_sudoku}")
    return render_template("next.html", sudoku=next_sudoku)

@app.route("/solution", methods=["POST"])
def solution():
    """
    This view shows an alert of wether the sudoku 
    was solved correctly or not. Maybe I should add a
    button that POSTs to a /next page.
    """

    # TODO: fix this after implementing the database
    # logic.

    # Get the time it took to solve the puzzle
    # I have to do this if people refresh the /solution
    # page. (?)
    if session["final"] is None:
        session["final"] = time.time()

    time_it_took = session["final"] - session["start"]
    print(f"it took {time_it_took}")
    
    # Should I reset start?
    # No. Start should get reset when the player
    # starts a new sudoku.
    # session["start"] = None

    # Get the board from the request
    data = request.form
    board = np.array(parse_data(data))
    solved, message = check_solution(board)

    if solved:
        # Registering the time
        print("Registering time.")
        se = SudokuExperiment.from_json(session["se"])
        se.register_time(time_it_took)

        # Plotting the mean of the GP and the acquisition for debugging
        print("Visualizing.")
        se.visualize()

        # Saving after fitting the GP
    else:
        # Ignore that last hint
        session["se"]["hints"].pop(-1)
        assert len(session["se"]["hints"]) == len(session["se"]["times"])
    
    session["se"] = se.to_json()

    # TODO: implement what happens if not solved.
    return render_template(
        "solution.html",
        solved=solved,
        message=message,
        time_it_took=int(time_it_took))


@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    print("Serving the web app")
    app.run(debug=True)
