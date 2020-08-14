"""
Creating the DB and the tables.
"""
import sqlite3
import json

from sudoku_utilities import sudoku_to_string

with open("config.json") as fp:
    config = json.load(fp)

class Models:
    def __init__(self, db):
        """
        Takes a db connection.
        """
        self.db = db
    
    def execute_query(self, query):
        c = self.db.cursor()
        c.execute(query)
        self.db.commit()

    def create_experiments_table(self):
        query = "CREATE TABLE IF NOT EXISTS experiments ("
        query += "exp_id INT,"
        query += "goal INT"
        query += ")"
        self.execute_query(query)

    def create_sudokus_table(self):
        query = "CREATE TABLE IF NOT EXISTS sudokus ("
        query += "exp_id INT,"
        query += "sudoku TEXT,"
        query += "hint INT,"
        query += "start_time FLOAT"
        query += ")"
        self.execute_query(query)

    def create_solutions_table(self):
        query = "CREATE TABLE IF NOT EXISTS solutions ("
        query += "exp_id INT,"
        query += "sudoku_solved TEXT,"
        query += "finish_time FLOAT,"
        query += "solved BIT"
        query += ")"
        self.execute_query(query)

    def is_this_experiment_in(self, exp_id):
        query = f"SELECT * FROM experiments WHERE exp_id={exp_id}"
        c = self.db.cursor()
        c.execute(query)
        experiments = c.fetchall()
        if len(experiments) == 0:
            return False
        else:
            assert len(experiments) == 1
            return True

    def get_experiments_table(self):
        query = "SELECT * FROM experiments"
        c = self.db.cursor()
        c.execute(query)
        experiments = c.fetchall()
        return experiments

    def get_sudokus_table(self):
        query = "SELECT * FROM sudokus"
        c = self.db.cursor()
        c.execute(query)
        experiments = c.fetchall()
        return experiments

    def get_solutions_table(self):
        query = "SELECT * FROM solutions"
        c = self.db.cursor()
        c.execute(query)
        experiments = c.fetchall()
        return experiments


    def save_experiment(self, exp_id, goal):
        query = "INSERT INTO experiments VALUES ("
        query += f"{exp_id},"
        query += f"{goal});"
        self.execute_query(query)

    def save_sudoku(self, exp_id, sudoku, hint, start):
        sudoku_as_string = sudoku_to_string(sudoku)
        query = "INSERT INTO sudokus VALUES ("
        query += f"{exp_id},"
        # query += "s" + sudoku_as_string + ","
        query += "\"" + sudoku_as_string + "\","
        query += f"{hint},"
        query += f" {start});"
        self.execute_query(query)

    def save_solution(self, exp_id, sudoku_solved, finish, solved):
        sudoku_as_string = sudoku_to_string(sudoku_solved)
        print(sudoku_as_string)
        query = "INSERT INTO solutions VALUES ("
        query += f"{exp_id},"
        # query += "s" + sudoku_as_string + ","
        query += "\"" + sudoku_as_string + "\","
        query += f" {finish},"
        query += f" {1 if solved else 0});"
        print(query)
        self.execute_query(query)

if __name__ == "__main__":
    db = sqlite3.connect("data.db")
    models = Models(db)
    models.create_experiments_table()
    models.create_sudokus_table()
    models.create_solutions_table()
