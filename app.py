import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import sqlite3
from flask import Flask, render_template, url_for, request, session

from sudoku_utilities import sudoku_to_string
from solution_checking import parse_data, check_solution
from sudoku_experiment import SudokuExperiment

from models import Models

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
    # Assign a random id to the player.
    if "user_id" not in session:
        session["user_id"] = str(time.time()).replace(".", "")

    goal = 30
    session["goal"] = goal
    se = SudokuExperiment(
        goal,
        name=f"{session['user_id']}"
        # debugging=True
    )
    session["se"] = se.to_json()
    session["start"] = None

    return render_template("index.html")

@app.route("/next")
def next():
    se = SudokuExperiment.from_json(session["se"])
    session["start"] = time.time()
    session["final"] = None

    # This operation stores one hint in self.hints.
    next_sudoku = se.next_sudoku()
    session["se"] = se.to_json()
    session["next_sudoku"] = next_sudoku
    print(f"next sudoku: {next_sudoku}")
    return render_template("next.html", sudoku=next_sudoku)

@app.route("/solution", methods=["POST"])
def solution():
    """
    TODO: write this
    """
    # Get the time it took to solve the puzzle
    # I have to do this if people refresh the /solution
    # page. (?)
    if session["final"] is None:
        # Then the player is accessing this page
        # for the first time.
        update = True
        session["final"] = time.time()
        time_it_took = session["final"] - session["start"]
        print(f"it took {time_it_took}")
    else:
        print("Did you just reload the page?")
        update = False
        session["final"] = min(
            (time.time(), session["final"])
        )
        time_it_took = session["final"] - session["start"]
        print(f"it took {time_it_took}")

    # Get the board from the request
    data = request.form
    board = np.array(parse_data(data))
    solved, message = check_solution(board)

    # Save this whole thing into the db.
    print("Saving in the database.")
    db = sqlite3.connect(config["DATABASE"])
    m = Models(db)
    m.save_trial(
        session["user_id"],
        sudoku_to_string(session["next_sudoku"]),
        solved,
        time_it_took
    )
    db.close()

    se = SudokuExperiment.from_json(session["se"])
    if solved:
        if update:
            # Registering the time
            print("Registering time.")
            # Here, it's trying to fit the GP.
            se.register_time(time_it_took)

            # Plotting the mean of the GP and the acquisition for debugging
            print("Visualizing.")
            se.visualize()
        else:
            print("We're not updating. Did you just reload the page?")
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
