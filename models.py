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
        query += "sudoku varchar(81),"
        query += "hint INT,"
        query += "start_time FLOAT"
        query += ")"
        self.execute_query(query)

    def create_solutions_table(self):
        query = "CREATE TABLE IF NOT EXISTS solutions ("
        query += "exp_id INT,"
        query += "sudoku_solved varchar(81),"
        query += "finish_time FLOAT,"
        query += "solved BIT"
        query += ")"
        self.execute_query(query)
    
    def save_sudoku(self, exp_id, sudoku, hint, start):
        query = "INSERT INTO sudokus VALUES ("
        query += f"{exp_id},"
        query += f"{sudoku_to_string(sudoku)},"
        query += f"{hint},"
        query += f" {start});"
        self.execute_query(query)

    def save_solution(self, exp_id, sudoku_solved, finish, solved):
        query = "INSERT INTO solutions VALUES ("
        query += f"{exp_id},"
        query += f"{sudoku_to_string(sudoku_solved)},"
        query += f" {finish},"
        query += f" {1 if solved else 0});"
        self.execute_query(query)

if __name__ == "__main__":
    db = sqlite3.connect("data.db")
    models = Models(db)
    models.create_experiments_table()
    models.create_sudokus_table()
    models.create_solutions_table()
