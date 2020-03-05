import sqlite3
import os
from pathlib import Path
from datetime import datetime

project_directory = r'D:\Projects\CircleTrack'
db_path = os.path.join(project_directory, 'mice.sqlite')


def make_db(db_path=db_path):
    """
    Makes an empty database with the following tables.

    mouse:
        name TEXT: Name of the mouse.

    sessions:
        mouse_id INTEGER: ID # of the mouse, references the
            primary key of the "mouse" table.
        exp_date DATE: Date the session was recorded.
        exp_time DATETIME: Time the session was recorded.
        path: Path to the session's folder.

    webcam:
        session_id INTEGER: References the primary key of the
            "sessions" table.
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
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mouse
            (id INTEGER PRIMARY KEY, 
            name TEXT UNIQUE)
            """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions
            (id INTEGER PRIMARY KEY,
            mouse_id INTEGER,
            datetime DATETIME,
            path TEXT UNIQUE,
            UNIQUE(mouse_id, path))
            """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS webcam
            (id INTEGER PRIMARY KEY,
            session_id INTEGER,
            vid_path TEXT,
            eztrack_path TEXT,
            dlc_path TEXT,
            UNIQUE(session_id, vid_path, eztrack_path, dlc_path))
            """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS miniscope
            (id INTEGER PRIMARY KEY,
            session_id INTEGER,
            minian_path TEXT,
            UNIQUE(session_id, minian_path))
            """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS licking
            (id INTEGER PRIMARY KEY, 
            session_id INTEGER,
            txt_path TEXT,
            UNIQUE(session_id, txt_path))
            """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS aggregate
            (id INTEGER PRIMARY KEY,
            session_id INTEGER,
            Preprocess_path TEXT,
            UNIQUE(session_id, Preprocess_path))
            """)

def update_db(directory=project_directory, db_path=db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # Update database with mice.
        mouse_folders = update_mice(cur, directory)

        # Update database with sessions.
        update_sessions(cur, mouse_folders)


        conn.commit()


def update_mice(cur, directory):
    mouse_folders = [f.path for f in os.scandir(directory) if f.is_dir()]
    mouse_names = [(os.path.split(mouse)[-1],) for mouse in mouse_folders]
    cur.executemany('''
        INSERT OR IGNORE INTO mouse (name) 
        VALUES (?)''', (mouse_names))

    return mouse_folders


def update_sessions(cur, mouse_folders):
    for mouse_id, mouse_folder in enumerate(mouse_folders):
        data = [(mouse_id,
                 datetime.strptime(folder.parts[-2] + ' ' +
                                   folder.parts[-1],
                                   '%m_%d_%Y H%H_M%M_S%S'),
                 folder._str) for folder in
                Path(mouse_folder).rglob('H*_M*_S*')
                if folder.is_dir()]

        cur.executemany('''
            INSERT OR IGNORE INTO sessions (mouse_id, datetime, path)
            VALUES (?,?,?)''', data)


#def update_sessions(cur, directory):

if __name__ == '__main__':
    make_db()
    update_db()