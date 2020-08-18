class Trials:
    """
    The trials table
    """

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
        query += "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        query += "user_id VARCHAR(36),"
        query += "sudoku TEXT,"
        query += "solved BIT,"
        query += "took FLOAT"
        query += ")"
        self.execute_query(query)

    def get_solved_for_user(self, user_id):
        query = f"SELECT * FROM trials WHERE user_id='{user_id}' AND solved=1"
        c = self.db.cursor()
        c.execute(query)
        trials = c.fetchall()
        return trials

    def save_trial(self, user_id, sudoku, solved, took):
        query = f"INSERT INTO trials (user_id,sudoku,solved,took) VALUES ('{user_id}', '{sudoku}', {1 if solved else 0}, {took});"
        print(query)
        self.execute_query(query)
