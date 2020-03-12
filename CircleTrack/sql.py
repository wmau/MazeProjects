import sqlite3
import os
from pathlib import Path
from datetime import datetime

project_directory = r'D:\Projects\CircleTrack'

class Database:
    def __init__(self, directory=project_directory, db_name='mice.sqlite',
                 vid_fname='Merged.avi',
                 eztrack_pattern='*_LocationOutput.csv',
                 DLC_pattern='*DLC_resnet*.h5',
                 minian_folder='minian',
                 lick_pattern='H*_M*_S*.txt',
                 preprocess_fname='PreprocessedBehavior.csv'):
        self.directory = directory
        self.db_path = os.path.join(directory, db_name)

        # Store the file name patterns.
        self.vid_fname = vid_fname
        self.eztrack_pattern = eztrack_pattern
        self.DLC_pattern = DLC_pattern
        self.minian_folder = minian_folder
        self.lick_pattern = lick_pattern
        self.preprocess_fname = preprocess_fname

        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()


    def __enter__(self):
        return self


    def __del__(self):
        self.connection.close()


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        if isinstance(exc_val, Exception):
            self.connection.rollback()
        else:
            self.connection.commit()

        self.connection.close()


    def make_db(self):
        """
        Makes an empty database with the following tables.

        mouse:
            name TEXT: Name of the mouse.

        session:
            mouse_id INTEGER: ID # of the mouse, references the
                primary key of the "mouse" table.
            exp_date DATE: Date the session was recorded.
            exp_time DATETIME: Time the session was recorded.
            path: Path to the session's folder.

        webcam:
            session_id INTEGER: References the primary key of the
                "session" table.
            vid_path TEXT: Path to the video file.
            eztrack_path TEXT: Path to the ezTrack csv.
            dlc_path TEXT: Path to the DeepLabCut h5.

        miniscope:
            session_id INTEGER: References the primary key of the
                "sessions" table.
            minian_path TEXT: Path to minian folder.

        licking:
            session_id INTEGER: References the primary key of the
                "sessions" table.
            txt_path TEXT: Path to Arduino txt file containing licking
                timestamps.

        aggregate:
            session_id INTEGER: References the primary key of the
                "sessions" table.
            Preprocess_path: Path to the PreprocessedData.csv.

        :param db_path:
        :return:
        """

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS mouse
            (id INTEGER PRIMARY KEY, 
            name TEXT UNIQUE)
            """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session
            (id INTEGER PRIMARY KEY,
            mouse_id INTEGER,
            datetime DATETIME,
            path TEXT UNIQUE,
            UNIQUE(mouse_id, path))
            """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS webcam
            (id INTEGER PRIMARY KEY,
            session_id INTEGER,
            vid_path TEXT,
            eztrack_path TEXT,
            dlc_path TEXT,
            UNIQUE(session_id, vid_path, eztrack_path, dlc_path))
            """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS miniscope
            (id INTEGER PRIMARY KEY,
            session_id INTEGER,
            minian_path TEXT,
            UNIQUE(session_id, minian_path))
            """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS licking
            (id INTEGER PRIMARY KEY, 
            session_id INTEGER,
            txt_path TEXT,
            UNIQUE(session_id, txt_path))
            """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregate
            (id INTEGER PRIMARY KEY,
            session_id INTEGER,
            Preprocess_path TEXT,
            UNIQUE(session_id, Preprocess_path))
            """)


    def update_db(self):
        self.mouse_folders = \
            [f.path for f in os.scandir(self.directory)
             if f.is_dir()]

        # Update database with mice.
        self.update_mice()
        self.connection.commit()

        # Update database with sessions.
        self.update_sessions()
        self.connection.commit()

        # Update database with behavior data.
        self.update_behavior()
        self.connection.commit()

        # Update database with miniscope data.
        self.update_miniscope()
        self.connection.commit()

        # Update database with licking data.
        self.update_licking()
        self.connection.commit()

        # Update database with preprocessed data.
        self.update_preprocess()
        self.connection.commit()


    def update_mice(self):
        mouse_names = [(os.path.split(mouse)[-1],)
                       for mouse in self.mouse_folders]
        self.cursor.executemany('''
            INSERT OR IGNORE INTO mouse (name) 
            VALUES (?)''', (mouse_names))


    def update_sessions(self):
        # Lambda function for formatting date time.
        format_datetime = lambda folder: \
            datetime.strptime(folder.parts[-2] + ' ' + folder.parts[-1],
                              '%m_%d_%Y H%H_M%M_S%S')

        self.session_folders = []
        # For each mouse, capture the datetime and directory.
        for mouse_id, mouse_folder in enumerate(self.mouse_folders):
            data = [(mouse_id + 1,
                     format_datetime(folder),
                     str(folder)) for folder in
                    Path(mouse_folder).rglob('H*_M*_S*')
                    if folder.is_dir()]

            self.cursor.executemany('''
                INSERT OR IGNORE INTO session
                (mouse_id, datetime, path)
                VALUES (?,?,?)''', data)

            self.session_folders.extend([Path(session[-1]) for session in data])


    def update_behavior(self):
        for session_folder in self.session_folders:
            session_id = self.id_session(session_folder)

            vid_path = str(session_folder / self.vid_fname)

            try:
                eztrack_path = [str(x) for x in session_folder.rglob(self.eztrack_pattern)][0]
            except:
                eztrack_path = None

            try:
                dlc_path = [str(x) for x in session_folder.rglob(self.DLC_pattern)][0]
            except:
                dlc_path = None

            self.cursor.execute('''
                INSERT OR IGNORE INTO webcam 
                (session_id, vid_path, eztrack_path, dlc_path)
                VALUES (?,?,?,?)''',
                (session_id, vid_path, eztrack_path, dlc_path))


    def update_miniscope(self):
        for session_folder in self.session_folders:
            session_id = self.id_session(session_folder)

            minian_path = session_folder / self.minian_folder
            if not minian_path.exists():
                minian_path = None

            self.cursor.execute('''
                INSERT OR IGNORE INTO miniscope
                (session_id, minian_path)
                VALUES (?,?)''',
                (session_id, str(minian_path)))


    def update_licking(self):
        for session_folder in self.session_folders:
            session_id = self.id_session(session_folder)

            try:
                lick_path = [str(x) for x in session_folder.rglob(self.lick_pattern)][0]
            except:
                lick_path = None

            self.cursor.execute('''
                INSERT OR IGNORE INTO licking
                (session_id, txt_path)
                VALUES (?,?)''',
                (session_id, lick_path))


    def update_preprocess(self):
        for session_folder in self.session_folders:
            session_id = self.id_session(session_folder)

            try:
                preprocess_path = [str(x) for x in session_folder.rglob(self.preprocess_fname)][0]
            except:
                preprocess_path = None

            self.cursor.execute('''
                   INSERT OR IGNORE INTO aggregate
                   (session_id, Preprocess_path)
                   VALUES (?,?)''',
                   (session_id, preprocess_path))


    def id_session(self, session_folder):
        self.cursor.execute('''
            SELECT id FROM session
            WHERE path = ? LIMIT 1''', (str(session_folder),))

        return self.cursor.fetchone()[0]


    def conditional_query(self, table, wanted_column, column, condition):
        """
        Lets you find the path or ID # of mice or sessions.

        :parameters
        ---
        table: str
            Table you want to retrieve data from (e.g., 'session').

        wanted_column: str
            Column of table you want to retrieve data from (e.g., 'path').

        column: str
            Column you are conditioning on (e.g., 'mouse_id').

        condition: str, tuple
            Condition (e.g., 'Mouse4').
        """
        if not isinstance(condition, tuple):
            condition = (condition,)

        query = f'''
            SELECT {wanted_column} FROM {table} 
            WHERE {column} = ?'''
        self.cursor.execute(query, condition)

        results = self.cursor.fetchall()
        return [result[0] for result in results]

if __name__ == '__main__':
    with Database() as db:
        db.make_db()
        db.update_db()
        #mouse_id = db.conditional_ID_query('mouse', 'id', 'name', 'Mouse4')[0]
        #i = db.conditional_ID_query('session', 'path', 'mouse_id', mouse_id)
        pass