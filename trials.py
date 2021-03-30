import psycopg2

class Trials:
    """
    The trials table
    """

    def __init__(self, db):
        """
        Takes a db connection.
        """
        self.table_name = "trials_regression"
        self.db = db

    def execute_query(self, query):
        c = self.db.cursor()
        c.execute(query)
        self.db.commit()

    def create_trials_table(self):
        query = f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
        query += "id SERIAL PRIMARY KEY,"
        query += "user_id VARCHAR(36),"
        query += "sudoku VARCHAR(81),"
        query += "solved BOOLEAN,"
        query += "took FLOAT"
        query += ")"
        try:
            self.execute_query(query)
        except psycopg2.errors.UniqueViolation:
            print("Table already exists?")
            pass

    def get_solved_for_user(self, user_id):
        query = f"SELECT * FROM {self.table_name} WHERE user_id='{user_id}' AND solved=TRUE"
        c = self.db.cursor()
        c.execute(query)
        trials = c.fetchall()
        return trials

    def save_trial(self, user_id, sudoku, solved, took):
        query = f"INSERT INTO {self.table_name} (user_id,sudoku,solved,took) VALUES ('{user_id}', '{sudoku}', {'TRUE' if solved else 'FALSE'}, {took});"
        print(query)
        self.execute_query(query)
