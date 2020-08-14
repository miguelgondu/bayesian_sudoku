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
    exp_id = str(time.time()).replace(".", "")
    session["experiment_id"] = exp_id
    print(f"Got experiment id: {session['experiment_id']}")

    print("Creating the Sudoku Experiment object")
    goal = 30
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
    se = SudokuExperiment.from_json(session["se"])
    session["start"] = time.time()
    session["final"] = None

    # This operation stores one hint in self.hints.
    next_sudoku = se.next_sudoku()
    session["se"] = se.to_json()

    print("Saving in the database.")
    db = sqlite3.connect("data.db")
    m = Models(db)
    # Check if this experiment has been saved in
    # the experiments table.
    if not m.is_this_experiment_in(session["experiment_id"]):
        m.save_experiment(
            session["experiment_id"],
            session["se"]["goal"]
        )
    print(np.array(next_sudoku))
    print(np.where(np.array(next_sudoku) == 0)[0])
    print(f"Empty spots: {len(np.where(np.array(next_sudoku) == 0)[0])}")

    m.save_sudoku(
        session["experiment_id"],
        np.array(next_sudoku),
        81 - len(np.where(np.array(next_sudoku) == 0)[0]),
        session["start"]
    )
    db.close()

    print(f"next sudoku: {next_sudoku}")
    print(session)
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
    if update:
        print("Saving in the database.")
        db = sqlite3.connect("data.db")
        m = Models(db)
        m.save_solution(
            session["experiment_id"],
            board,
            session["final"],
            solved
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
            # print("Visualizing.")
            # se.visualize()
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
