import os
from pathlib import Path
import pickle as pkl
import tkinter as tk
import re

from CaImaging.util import get_data_paths

tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog
import pandas as pd
import numpy as np


def make_pattern_dict():
    """
    Makes the dictionary that tells get_data_paths() where each data
    file lives.

    :return:
    ---
    pattern_dict: dict
        Dictionary where fields are
    """
    pattern_dict = {
        'Arduino': '^H\d{2}_M\d{2}_S\d{2}.\d{4} \d{4}.txt$',
        'BehaviorVideo': 'Merged.avi',
        'DLC': '.*DLC_resnet.*.h5',
        'BehaviorData': '.*_LocationOutput.csv',
        'timestamps': '^time[s,S]tamp',
        'PreprocessedBehavior': 'PreprocessedBehavior.csv',
        'minian': '^minian$'
    }

    return pattern_dict


def grab_paths(session_folder=None):
    """
    Get the data paths for a session folder.

    :param session_folder:
    :return:
    """
    pattern_dict = make_pattern_dict()

    if session_folder is None:
        session_folder = filedialog.askdirectory()

    paths = get_data_paths(session_folder, pattern_dict)

    return paths


class Session_Metadata:
    def __init__(self, session_folder=None, overwrite=False):
        """
        Locate the metadata in a session. This will typically
        include things like minian output folder, behavior
        tracking csvs, timestamps, etc. The files found
        are defined by the pattern dict in the function
        make_pattern_dict().

        :parameters
        ---
        session_folder: str
            Full path to the folder you want to extract
            metadata from.

        overwrite: bool
            Whether or not to overwrite the existing
            metadata csv.

        """
        # If folder is not specified, open a dialog box.
        if session_folder is None:
            self.session_folder = filedialog.askdirectory()
        else:
            self.session_folder = session_folder

        # Get full file path to the metadata csv.
        self.full_path = os.path.join(session_folder, 'metadata.pkl')

        if overwrite:
            self.build()
            self.save()
            self.meta_dict = self.load()
        else:
            try:
                self.meta_dict = self.load()
            except:
                self.build()
                self.save()
                self.meta_dict = self.load()


    def build(self):
        """
        Gather all the paths.

        """
        self.filepaths = grab_paths(self.session_folder)


    def save(self):
        """
        Pickle the dict to disk. .


        """
        with open(self.full_path, 'wb') as file:
            pkl.dump(self.filepaths, file)


    def load(self):
        """
        Load pickled dict.

        :return:
        """
        with open(self.full_path, 'rb') as file:
            meta_dict = pkl.load(file)

        return meta_dict



class Metadata_CSV:
    def __init__(self, folder=None, mouse=-3, date=-2, session=-1,
                 filename='Metadata.csv', overwrite=False):
        """
        Makes a CSV file containing the metadata of all the sessions
        for a particular project. This includes session folder
        locations as well as individual files within those folders.
        For example, ezTrack outputs or timestamps.dat.

        :param folder:
        """
        if folder is None:
            self.project_folder = filedialog.askdirectory()
        else:
            self.project_folder = folder
        self.filename = filename
        self.path_levels = {'mouse': mouse,
                            'date': date,
                            'session': session,
                            }

        fname = os.path.join(self.project_folder, filename)

        if overwrite:
            self.build()
            self.save()
            self.df = pd.read_csv(fname)
        else:
            try:
                self.df = pd.read_csv(fname)
            except:
                self.build()
                self.save()
                self.df = pd.read_csv(fname)


    def build(self):
        self.session_folders = self.get_all_sessions()
        for folder in self.session_folders:
            Session_Metadata(folder, overwrite=True)

        mouse_names = self.get_metadata('mouse')

        master_dict = {
            'Mouse': mouse_names,
            'Group': None, #to do
            'Session': self.get_metadata('date'),
            'Session_Type': self.get_session_type(),
            'Path': self.session_folders,
            'CellRegPath': [os.path.join(self.project_folder,
                                         mouse,
                                         'SpatialFootprints',
                                         'CellRegResults')
                            for mouse in mouse_names], #to do
            'Metadata': [os.path.join(folder, 'metadata.pkl')
                         for folder in self.session_folders]
        }

        self.df = pd.DataFrame(master_dict)


    def save(self):
        self.df.to_csv(os.path.join(self.project_folder,
                                    self.filename), index=False)


    def get_all_sessions(self):
        session_folders = []
        expression = '^H?[0-9]+_M?[0-9]+_S?[0-9]+$'
        for root, dirs, _ in os.walk(self.project_folder):
            for directory in dirs:
                if re.match(expression, directory):
                    session_folders.append(os.path.join(root, directory))

        # session_folders = [folder for folder in
        #                    Path(self.project_folder).rglob('H*_M*_S*')
        #                    if os.path.isdir(folder)]

        return session_folders


    def get_session_type(self):
        session_types = [folder.split(os.sep)[self.path_levels['date']].split('_')[-1]
                         for folder in self.session_folders]

        return session_types


    def get_metadata(self, path_level):
        """
        Get the metadata associated with each path.

        :parameter
        ---
        path_level: str
            'mouse', 'date', or 'session'
        """
        mice = [session.split(os.sep)[self.path_levels[path_level]]
                for session in self.session_folders]

        return mice

if __name__ == '__main__':
    #paths = grab_paths(r'Z:\Will\Drift\Data\Castor_Scope05\09_06_2020_CircleTrackShaping1\17_11_36')
    Metadata_CSV(r'Z:\Will\Drift\Data', overwrite=True)


