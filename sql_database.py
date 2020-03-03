import sqlite3
import os

folder = r'Z:\Will\Circle track'

def make_db(directory=folder):
    connection = sqlite3.connect(os.path.join(directory,
                                              'mice_desktop.sqlite'))

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mouse
        (id INTEGER PRIMARY KEY, 
        name TEXT,
        cage_card INTEGER)
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions
        (id INTEGER PRIMARY KEY,
        mouse_id INTEGER,
        exp_date DATE,
        exp_time DATETIME)
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS webcam
        (id INTEGER PRIMARY KEY,
        session_id INTEGER,
        vid_directory TEXT,
        eztrack_directory TEXT,
        dlc_directory TEXT)
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS miniscope
        (id INTEGER PRIMARY KEY,
        session_id INTEGER,
        vid_directory TEXT)
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS licking
        (id INTEGER PRIMARY KEY, 
        session_id INTEGER,
        directory TEXT)
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aggregate
        (id INTEGER PRIMARY KEY,
        session_id INTEGER,
        directory TEXT)
        """)

