import sqlite3
import os
from pathlib import Path
from datetime import datetime
from CaImaging.util import search_for_folders
import regex
import numpy as np
import pandas as pd

project_directory = r"D:\Projects\CircleTrack"
mouse_csv = r"Z:\Will\mouse_info.csv"

class Database:
    def __init__(self, directory=r"Z:\Will", db_name="database.sqlite",
                 from_scratch=False):
        self.directory = directory
        self.db_path = os.path.join(self.directory, db_name)

        if from_scratch:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)

        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        # Find all folders named 'Data'. The next folders inside should be mice.
        self.project_folders = search_for_folders(self.directory, "Data")
        self.path_levels = {
            "mouse": -3,
            "date": -2,
            "session": -1,
        }

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

    def create(self):
        self.make_db()
        self.populate_mice()
        self.populate_projects()
        self.populate_sessions()

    def execute(self, sql_str, tuple):
        output = self.cursor.execute(sql_str, tuple)

        return output.fetchall()

    def make_db(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mouse
            (id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            sex TEXT,
            dob DATETIME)
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS project
            (id INTEGER PRIMARY KEY,
            project_name TEXT,
            path TEXT UNIQUE)
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session
            (id INTEGER PRIMARY KEY,
            mouse_id INTEGER,
            project_id INTEGER,
            session_name TEXT, 
            datetime DATETIME,
            path TEXT UNIQUE,
            UNIQUE(mouse_id, path))
            """
        )

    def populate_projects(self):
        projects = [
            (i + 1, os.path.split(os.path.split(folder)[0])[-1], folder)
            for i, folder in enumerate(self.project_folders)
        ]

        self.cursor.executemany(
            """
            INSERT OR IGNORE INTO project (id, project_name, path)
            VALUES (?,?,?)""",
            (projects),
        )

    def populate_mice(self):
        mouse_info = pd.read_csv(mouse_csv)
        for project in self.project_folders:
            mice = [
                (os.path.split(f.path)[-1],) for f in os.scandir(project) if f.is_dir()
            ]

            mice = [
                tuple(mouse_info[mouse_info['Name'] == mouse[0]].values[0]) for mouse in mice
            ]

            self.cursor.executemany(
                """
                INSERT OR IGNORE INTO mouse (name, sex, dob)
                VALUES (?, ?, ?)""",
                (mice),
            )

    def populate_sessions(self):
        # Turn datetime string into a datetime variable.
        format_datetime = lambda folder: datetime.strptime(
            self.extract_folder_info(folder)[0], "%m_%d_%Y %H_%M_%S"
        )

        for project_id, project in enumerate(self.project_folders):
            mouse_folders = [f.path for f in os.scandir(project) if f.is_dir()]

            for mouse_folder in mouse_folders:
                mouse_name = os.path.split(mouse_folder)[-1]
                mouse_id = self.cursor.execute(
                    """
                    SELECT id
                    FROM mouse
                    WHERE name = ?""",
                    (mouse_name,),
                ).fetchone()[0]

                data = [
                    (
                        mouse_id,
                        project_id + 1,
                        format_datetime(folder),
                        self.extract_folder_info(folder)[1],
                        folder,
                    )
                    for folder in search_for_folders(
                        mouse_folder, "^H?[0-9]+_M?[0-9]+_S?[0-9]+$"
                    )
                ]

                self.cursor.executemany(
                    """
                    INSERT OR IGNORE INTO session
                    (mouse_id, project_id, datetime, session_name, path)
                    VALUES (?,?,?,?,?)""",
                    data,
                )


    def extract_folder_info(self, folder):
        """
        Takes a file path and extracts the date, time, and session type.
        For example, ~\02_10_2021_CircleTrackReversal1\15_24_41 will be
        placed into the tuple: ('02_10_2021 15_24_41', 'CircleTrackReversal1').

        :parameter
        ---
        folder: str
            Full path to data folder. Assumes the following structure:
            ~\MM_DD_YYYY_SessionType\HH_MM_SS
            OR
            ~\YYYY_MM_DD_SessionType\HH_MM_SS
            The timestamp folder (HH_MM_SS) could also have H, M, and S
            in the folder name (e.g., H15_M24_S41). And there can also
            be either 1 or 2 digits.
        """
        # Assumes timestamp folder is the deepest folder in the path.
        timestamp_folder = os.path.split(folder)[-1]
        ts_matches = regex.findall("[0-9]+", timestamp_folder)

        # Assumes the date folder is the next deepest folder in the path.
        date_folder = os.path.split(os.path.split(folder)[0])[-1]
        date_matches = regex.findall("[0-9]{2,4}", date_folder)
        date_matches.sort(key=len)

        # Build the string.
        datetime_str = (
            f"{date_matches[0]}_{date_matches[1]}_{date_matches[-1]} "
            f"{ts_matches[0]}_{ts_matches[1]}_{ts_matches[2]}"
        )

        # Get the end of the date string and then parse the rest of
        # the date folder into the session_type string.
        end_of_date_idx = regex.match(
            "[0-9]{2,4}_[0-9]{2}_[0-9]{2,4}_", date_folder
        ).end()
        session_type = date_folder[end_of_date_idx:]

        return datetime_str, session_type


if __name__ == "__main__":
    with Database(from_scratch=True) as db:
        db.create()
