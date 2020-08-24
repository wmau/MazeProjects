import os
import glob
from pathlib import Path
import tkinter as tk

from CaImaging.util import get_data_paths
from CaImaging.Behavior import convert_dlc_to_eztrack

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
        'Arduino': 'H??_M??_S??.???? ????.txt',
        'BehaviorVideo': 'Merged.avi',
        'DLC': '*DLC_resnet*.h5',
        'BehaviorData': '*_LocationOutput.csv',
        'settings': 'settings_and_notes.dat',
        'timestamps': 'timestamp.dat'
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


class Metadata_CSV:
    def __init__(self, folder=None, mouse=-3, date=-2, session=-1,
                 filename='Metadata.csv'):
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

        try:
            self.df = pd.read_csv(os.path.join(self.project_folder,
                                               filename))

        except:
            self.session_folders = self.get_all_sessions()
            self.files_per_folder = [grab_paths(folder)
                                     for folder in self.session_folders]

            master_dict = {
                'Mouse': self.get_metadata('mouse'),
                'Group': None,
                'Session': self.get_metadata('date'),
                'Path': self.session_folders,
                'CellRegPath': None,
                'MiniscopeCam': None,
                'BehaviorCam': None,
                'BehaviorVideo': [files['BehaviorVideo']
                                  for files in self.files_per_folder],
                'Timestamps': [files['timestamps']
                               for files in self.files_per_folder],
                'BehaviorData': self.resolve_behavior_data()
                           }

            self.df = pd.DataFrame(master_dict)
            self.save()



    def save(self):
        self.df.to_csv(os.path.join(self.project_folder,
                                    self.filename), index=False)

    def get_all_sessions(self):
        session_folders = [folder for folder in
                           Path(self.project_folder).rglob('H*_M*_S*')
                           if os.path.isdir(folder)]

        return session_folders


    def get_metadata(self, path_level):
        """
        Get the metadata associated with each path.

        :parameter
        ---
        path_level: str
            'mouse', 'date', or 'session'
        """
        mice = [session._parts[self.path_levels[path_level]]
                for session in self.session_folders]

        return mice


    def resolve_behavior_data(self):
        """
        Some session will have ezTrack-analyzed csvs and others will
        have DeepLabCut-analyzed h5s. This function picks one or the
        other to be written to the Metadata csv. If the session has
        DLC-analyzed data, first convert it to a csv.

        :return:
        """
        behavior_files = []
        for paths in self.files_per_folder:

            # Keep this order in the if statement.
            if paths['BehaviorData']:
                behavior_files.append(paths['BehaviorData'])
            elif paths['DLC']:
                behavior_file = convert_dlc_to_eztrack(paths['DLC'])[1]
                behavior_files.append(behavior_file)
            else:
                print('No analyzed behavior found. '
                      'Run ezTrack or DeepLabCut')
                behavior_files.append(None)

        return behavior_files


    def update(self, session_folder):
        idx = self.df['Path'] == session_folder
        entry_match = self.df.loc[idx]

        new_paths = grab_paths(entry_match['Path'].values[0])

        try:
            behavior_path = convert_dlc_to_eztrack(new_paths['DLC'])[1]
        except:
            behavior_path =  new_paths['BehaviorData']

        self.df.loc[idx, 'BehaviorData'] = behavior_path

        self.save()

if __name__ == '__main__':
    project_folder = r'Z:\Will\Drift\Data'
    M = Metadata_CSV(project_folder)
    M.update(r'Z:\Will\Drift\Data\M1\07_11_2020_TMazeFreeChoice1\H14_M46_S13')


