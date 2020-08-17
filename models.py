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

    def create_trials_table(self):
        query = "CREATE TABLE IF NOT EXISTS trials ("
        query += " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        query += "user_id INT,"
        query += "sudoku TEXT,"
        query += "solved BIT,"
        query += "took FLOAT"
        query += ")"
        self.execute_query(query)

    def is_this_trial_in(self, _id):
        query = f"SELECT * FROM trials WHERE id={_id}"
        c = self.db.cursor()
        c.execute(query)
        experiments = c.fetchall()
        if len(experiments) == 0:
            return False
        else:
            assert len(experiments) == 1
            return True

    def get_trials_table(self):
        query = "SELECT * FROM trials"
        c = self.db.cursor()
        c.execute(query)
        experiments = c.fetchall()
        return experiments

    def save_trial(self, user_id, sudoku, solved, took):
        query = "INSERT INTO trials (user_id,sudoku,solved,took) VALUES ("
        query += f"{user_id},"
        query += "\"" + f"{sudoku}" + "\","
        query += f"{1 if solved else 0},"
        query += f"{took});"
        print(query)
        self.execute_query(query)

if __name__ == "__main__":
    db = sqlite3.connect(config["DATABASE"])
    models = Models(db)
    models.create_trials_table()
