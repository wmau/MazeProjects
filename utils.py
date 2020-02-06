import os
from util import get_data_paths, concat_avis
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog
from pathlib import Path
from Behavior import convert_dlc_to_eztrack

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
    folders = [folder for folder in Path(mouse_folder).rglob('H??_M??_S??')]

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



def dlc_to_csv(folder):
    paths = grab_paths(folder)
    data = convert_dlc_to_eztrack(paths['DLC'])

    return data

if __name__ == '__main__':
    path = r'Z:\Will\Circle track pilots\Mouse4'
    batch_concat_avis(path)