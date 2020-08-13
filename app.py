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
    size = 9
    goal = 2 * 60
    se = SudokuExperiment(
        size,
        goal,
        name=f"{exp_id}",
        debugging=True
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
    
    # This operation stores the self.next_sudoku_ and self.next_hints
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
        # Fitting the GP
        print("Registering time.")
        se = SudokuExperiment.from_json(session["se"])
        se.register_time(time_it_took)
        session["se"] = se.to_json()

        # Plotting the mean of the GP and the acquisition for debugging
        # print("Visualizing.")
        # se.visualize()
    else:
        # TODO: remove the last hint, or re-plan how the hints are
        # being kept.
        # Implement what happens when the solution wasn't good.
        # What's that exactly?
        pass

    # TODO: implement what happens if not solved.
    return render_template("solution.html", solved=solved, message=message)


@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    print("Serving the web app")
    app.run(debug=True)
