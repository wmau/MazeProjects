import os
from util import get_data_paths, concat_avis, nan_array
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog
from pathlib import Path
from Behavior import convert_dlc_to_eztrack
import pandas as pd
from natsort import natsorted
from shutil import copyfile
import cv2
import re
from Miniscope import project_image

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
        'DLC': '*DLC_*_circletrack*_*.h5',
        'BehaviorData': '*_LocationOutput.csv',
        'settings': 'settings_and_notes.dat',
        'timestamps': 'timestamp.dat'
    }

    return pattern_dict


def circle_sizes(x, y):
    """
    Get the size of the circle track given visited x and y coordinates.
    The mouse must visit all parts of the circle.

    :parameters
    ---
    x, y: array-like
        x, y coordinates

    :return
    (width, height, radius, center): tuple
        Self-explanatory.
    """
    x_extrema = [min(x), max(x)]
    y_extrema = [min(y), max(y)]
    width = np.diff(x_extrema)[0]
    height = np.diff(y_extrema)[0]

    radius = np.mean([width, height])/2
    center = [np.mean(x_extrema), np.mean(y_extrema)]

    return (width, height, radius, center)


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


def cart2pol(x, y):
    """
    Cartesian to polar coordinates. For linearizing circular trajectory.

    :parameters
    ---
    x, y: array-like
        x, y coordinates

    :return
    ---
    (phi, rho): tuple
        Angle (linearized distance) and radius (distance from center).
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return (phi, rho)


def batch_concat_avis(mouse_folder):
    """
    Batch concatenates the avi chunks in a mouse folder.

    :parameter
    ---
    mouse_folder: str
        Directory containing session folders. The session folders
        must have the format H??_M??_S??.
    """
    # Recursively search for the session folders.
    folders = [folder for folder in Path(mouse_folder).rglob('H??_M*_S??')]

    # For each folder, check that Merged.avi doesn't already exist.
    for session in folders:
        merged_file = os.path.join(session, 'Merged.avi')

        if os.path.exists(merged_file):
            print(f'{merged_file} already exists')

        # If not, concatenate the avis.
        else:
            try:
                concat_avis(session, pattern='behavCam*.avi',
                            fname='Merged.avi', fps=30)
            except:
                print(f'Failed to create {merged_file}')



def dlc_to_csv(folder: str):
    """
    Finds the DLC output file and converts it to csv, mirroring
    the format of ezTrack outputs.

    :parameter
    ---
    folder: str
        Directory containing DLC output file (h5).

    :return
    ---
    data: DataFrame
        DLC data.
    """
    paths = grab_paths(folder)
    data = convert_dlc_to_eztrack(paths['DLC'])

    return data


def get_session_folders(mouse_folder: str):
    """
    Find all the session folders within a subtree under mouse_folder.

    :parameter
    ---
    mouse_folder: str
        Folder for a single mouse.

    :return
    ---
    folders: list of Paths
        Directories for each session.
    """
    folders = [folder for folder in Path(mouse_folder).rglob('H??_M*_S??')]

    return folders


class SessionStitcher:
    def __init__(self, folder_list, recording_duration=20,
                 miniscope_cam=6, behav_cam=2,
                 fps=30, pattern='msCam*.avi'):
        """
        Combine recording folders that were split as a result of
        the DAQ software crashing. Approach: In a new folder, copy
        all the miniscope files from the first folder, then add a
        file with empty frames that accounts for the time it took
        to reconnect. Then copy all the files from the second folder.
        For the behavior, do the same thing but merge them into one
        file.

        """

        self.folder_list = folder_list
        assert len(folder_list) == 2, "This only works for sessions with 1 crash."

        self.recording_duration = recording_duration
        self.miniscope_cam = miniscope_cam
        self.behav_cam = behav_cam
        self.fps = fps
        self.pattern = pattern

        self.missing_frames = self.calculate_missing_frames()
        self.stitched_folder = self.make_stitched_folder()
        self.copy_miniscope_files(self.folder_list[0], second=False)
        self.make_missing_data()
        self.copy_miniscope_files(self.folder_list[1], second=True)


    def calculate_missing_frames(self):
        # First, determine the number of missing frames.
        # Read the timestmap.dat files.
        timestamp_files = [os.path.join(folder, 'timestamp.dat')
                           for folder in self.folder_list]
        self.last_frames = []

        # Get the total number of frames for both session
        # folders.
        for file in timestamp_files:
            df = pd.read_csv(file, sep="\s+")

            self.last_frames.append(df.loc[df.camNum==self.miniscope_cam,
                                           'frameNum'].iloc[-1])

        # Calculate the discrepancy.
        recorded_frames = np.sum(self.last_frames)
        ideal_frames = self.recording_duration*60*self.fps
        missing_data_frames = ideal_frames - recorded_frames

        return missing_data_frames


    def make_stitched_folder(self):
        folder_name = self.folder_list[0] +  '_stitched'
        try:
            os.mkdir(folder_name)
        except:
            print(f'{folder_name} already exists.')

        return folder_name


    def get_miniscope_files(self, folder):
        files = natsorted([str(file) for file in Path(folder).rglob(self.pattern)])

        return files

    def copy_miniscope_files(self, source, second=False):
        files = self.get_miniscope_files(source)
        for file in files:
            fname = os.path.split(file)[-1]

            if second:
                current_number = int(re.findall(r'\d+', fname)[0])
                new_number = current_number + self.last_number
                fname = self.pattern.replace('*', str(new_number))

            destination = os.path.join(self.stitched_folder, fname)

            if os.path.isfile(destination):
                print(f'{destination} already exists. Skipping.')
            else:
                print(f'Copying {destination}.')
                copyfile(file, destination)


    def make_missing_data(self, frame='last'):
        files = self.get_miniscope_files(self.folder_list[0])
        last_video = files[-1]
        last_number = int(re.findall(r'\d+', os.path.split(last_video)[-1])[0])
        self.last_number = last_number + 1

        cap = cv2.VideoCapture(last_video)
        size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fname = self.pattern.replace('*', str(last_number))
        full_path = os.path.join(self.stitched_folder, fname)

        if not os.path.exists(full_path):
            video = cv2.VideoWriter(full_path, fourcc, float(self.fps), size)

            if frame == 'nan':
                frame = nan_array(size)
            elif frame == 'last':
                cap.set(1, cap.get(7)-1)
                ret, frame = cap.read()

            for _ in range(self.missing_frames):
                video.write(frame)

            video.release()
        cap.release()



if __name__ == '__main__':
    folder_list = [r'Z:\Will\Lingxuan_CircleTrack\03_05_2020\H14_M30_S5',
                   r'Z:\Will\Lingxuan_CircleTrack\03_05_2020\H14_M39_S37']
    SessionStitcher(folder_list)