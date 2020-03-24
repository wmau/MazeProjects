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
                 fps=30, miniscope_pattern='msCam*.avi',
                 behavior_pattern='behavCam*.avi'):
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
        self.fps = fps
        self.interval = int(np.round((1 / self.fps) * 1000))
        self.camNum = {'miniscope': miniscope_cam,
                               'behavior': behav_cam}
        self.file_patterns = {'miniscope': miniscope_pattern,
                              'behavior': behavior_pattern}
        self.timestamp_paths = [os.path.join(folder, 'timestamp.dat')
                                for folder in self.folder_list]

        # Find out how many frames are missing.
        self.missing_frames = self.calculate_missing_frames()
        self.stitched_folder = self.make_stitched_folder()

        self.merge_timestamp_files()
        self.stitch('miniscope')
        self.stitch('behavior')


    def stitch(self, camera):
        self.copy_files(self.folder_list[0], camera=camera, second=False)
        self.make_missing_data(camera=camera)
        self.copy_files(self.folder_list[1], camera=camera, second=True)


    def merge_timestamp_files(self):
        """
        Combine the timestamp.dat files by taking the first file,
        adding in missing data, then time-shifting the second file.

        """
        # Read the first session's timestamp file. Drop the last
        # entry since it's usually truncated.
        session1 = pd.read_csv(self.timestamp_paths[0], sep="\s+")
        session1.drop(session1.tail(1).index, inplace=True)

        missing_data = self.make_missing_timestamps(session1)

        session2 = pd.read_csv(self.timestamp_paths[1], sep="\s+")
        session2 = self.timeshift_second_session(missing_data, session2)

        # Merge.
        df = pd.concat((session1, missing_data, session2))
        path = os.path.join(self.stitched_folder, 'timestamp.dat')
        df = df.astype({'frameNum': int,
                        'sysClock': int,
                        'camNum': int})
        df.to_csv(path, sep='\t', index=False)


    def make_missing_timestamps(self, df):
        """
        Build the DataFrame with the missing timestamps.

        :parameter
        ---
        df: DataFrame
            DataFrame from first session.
        """
        # Define the cameras. We can loop through these to shorten the code.
        cameras = ['miniscope', 'behavior']

        # Find the last camera entry from session 1. Convert to str keys
        # used to access dicts.
        last_camera = df.camNum.iloc[-1]
        camera1 = [key for key, value in self.camNum.items() if value != last_camera][0]
        camera2 = [key for key, value in self.camNum.items() if value == last_camera][0]

        # Find the last frames and timestamps from session 1.
        last_frames = {camera: df.loc[df.camNum == self.camNum[camera], 'frameNum'].iloc[-1] + 1
                       for camera in cameras}
        last_ts = {camera: df.loc[df.camNum == self.camNum[camera], 'sysClock'].iloc[-1] + self.interval
                   for camera in cameras}

        # Build new frames and timestamps.
        frames = {camera: np.arange(last_frames[camera], last_frames[camera] + self.missing_frames)
                  for camera in cameras}
        ts = {camera: np.arange(last_ts[camera], last_ts[camera] + self.interval*self.missing_frames,
                                self.interval)
              for camera in cameras}

        # Make the DataFrame.
        data = dict()
        data['camNum'] = np.empty((self.missing_frames*2))
        data['frameNum'] = np.empty_like(data['camNum'])
        data['sysClock'] = np.empty_like(data['camNum'])

        for key, arr in zip(['camNum', 'frameNum', 'sysClock'],
                            [self.camNum, frames, ts]):
            data[key][::2] = arr[camera1]
            data[key][1::2] = arr[camera2]

        data = pd.DataFrame(data)
        data['buffer'] = 1

        return data


    def timeshift_second_session(self, missing_df, df2):
        """
        Shift the second session in time so that it lines up with the
        missing data.

        :parameters
        ---
        missing_df: DataFrame
            DataFrame from make_missing_timestamps()

        df2: DataFrame
            DataFrame from the second session.

        """
        cameras = ['miniscope', 'behavior']

        last_frames = {camera: missing_df.loc[missing_df.camNum==self.camNum[camera],
                                              'frameNum'].iloc[-1]
                       for camera in cameras}
        last_ts = missing_df.sysClock.iloc[-1]
        df2.sysClock += last_ts

        for camera in cameras:
            df2.loc[df2.camNum==self.camNum[camera], 'frameNum'] += last_frames[camera]

        # Correct that entry in sysClock that's really high at one of the
        # first two entries of timestamp.dat.
        i = np.argmax(df2.sysClock)
        spike_camera = df2.loc[i, 'camNum']
        spike_ts = df2.loc[df2.camNum==spike_camera, 'sysClock']
        next_ts = spike_ts.iloc[np.argmax(spike_ts) + 1]
        df2.loc[i, 'sysClock'] = next_ts - self.interval

        if df2.loc[0, 'sysClock'] > df2.loc[1, 'sysClock']:
            df2.loc[0, 'sysClock'] = df2.loc[1, 'sysClock'] - 1

        return df2

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

            self.last_frames.append(df.loc[df.camNum==self.camNum['miniscope'],
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


    def get_files(self, folder, pattern):
        files = natsorted([str(file) for file in Path(folder).rglob(pattern)])

        return files


    def copy_files(self, source, camera, second=False):
        pattern = self.file_patterns[camera]

        files = self.get_files(source, pattern)
        for file in files:
            fname = os.path.split(file)[-1]

            if second:
                current_number = int(re.findall(r'\d+', fname)[0])
                new_number = current_number + self.last_number
                fname = self.file_patterns[camera].replace('*', str(new_number))

            destination = os.path.join(self.stitched_folder, fname)

            if os.path.isfile(destination):
                print(f'{destination} already exists. Skipping.')
            else:
                print(f'Copying {destination}.')
                copyfile(file, destination)


    def make_missing_data(self, camera):
        pattern = self.file_patterns[camera]
        files = self.get_files(self.folder_list[0], pattern)
        last_video = files[-1]
        last_number = int(re.findall(r'\d+', os.path.split(last_video)[-1])[0])
        self.last_number = last_number + 1

        cap = cv2.VideoCapture(last_video)
        size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fname = self.file_patterns[camera].replace('*', str(self.last_number))
        full_path = os.path.join(self.stitched_folder, fname)

        if not os.path.exists(full_path):
            print(f'Writing {full_path}.')

            video = cv2.VideoWriter(full_path, fourcc, float(self.fps), size)

            cap.set(1, cap.get(7)-1)
            ret, frame = cap.read()

            for _ in range(self.missing_frames):
                video.write(frame)

            video.release()
        else:
            print(f'{full_path} already exists. Skipping.')

        cap.release()



if __name__ == '__main__':
    folder_list = [r'Z:\Will\Lingxuan_CircleTrack\03_05_2020\H15_M20_S43',
                   r'Z:\Will\Lingxuan_CircleTrack\03_05_2020\H15_M24_S58']
    SessionStitcher(folder_list)