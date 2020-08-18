import json
import psycopg2
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, session

from trials import Trials
from solution_checking import parse_data, check_solution
from sudoku_experiment import SudokuExperiment
from sudoku_utilities import sudoku_to_string

app = Flask(__name__)

# Squelch a warning
plt.switch_backend('Agg')

# These will need to get changed to os.environ
# when deploying with Heroku.
with open("config.json") as fp:
    config = json.load(fp)
secret_key = config["SECRET_KEY"]
app.secret_key = secret_key
db_url = config["DATABASE"]

db = psycopg2.connect(db_url)
trials = Trials(db)
trials.create_trials_table()
db.close()


@app.route("/")
def root():
    # Assign a random id to the player.
    if "user_id" not in session:
        session["user_id"] = uuid.uuid4()

    return render_template("index.html")


@app.route("/next")
def next():
    goal = 30

    db = psycopg2.connect(db_url)
    trials = Trials(db)
    solved = trials.get_solved_for_user(session["user_id"])
    hints = [81 - sudoku.count('0') for (_, _, sudoku, _, _) in solved]
    times = [took for (_, _, _, _, took) in solved]

    se = SudokuExperiment(
        goal,
        hints=hints,
        times=times,
        name=f"{session['user_id']}"
    )
    se.visualize()  # TODO this takes several seconds. Consider if we should do it offline.

    next_sudoku = se.next_sudoku()
    session["start"] = time.time()
    session["next_sudoku"] = next_sudoku
    print(f"next sudoku: {next_sudoku}")
    return render_template("next.html", sudoku=next_sudoku)


@app.route("/solution", methods=["POST"])
def solution():
    """
    Saves the trial in the db
    """

    time_it_took = time.time() - session["start"]
    data = request.form
    board = np.array(parse_data(data))
    solved, message = check_solution(board)

    # Save this whole thing into the db.
    print("Saving in the database.")
    db = psycopg2.connect("postgres://haqgglgbldkzch:a430a4e8455cd241225a454cfb40a4745a16c790aad049ee6c1e5088ee364be5@ec2-54-158-122-162.compute-1.amazonaws.com:5432/dcujjocm5f1qj3")
    trials = Trials(db)
    trials.save_trial(
        session["user_id"],
        sudoku_to_string(session["next_sudoku"]),
        solved,
        time_it_took
    )

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
