import os
from util import get_data_paths
import numpy as np
import matplotlib.pyplot as plt

def make_pattern_dict():
    miniscope_folder = 'H??_M??_S??'
    pattern_dict = {
        'Arduino': 'H??_M??_S??.???? ????.txt',
        'MiniscopeFolder': miniscope_folder,
        'BehaviorVideo': os.path.join(miniscope_folder, 'Merged.avi'),
        'ezTrack': os.path.join(miniscope_folder, '*_LocationOutput.csv'),
        'settings': os.path.join(miniscope_folder, 'settings_and_notes.dat'),
        'timestamps': os.path.join(miniscope_folder, 'timestamp.dat')
    }

    return pattern_dict


def circle_sizes(x, y):
    x_extrema = [min(x), max(x)]
    y_extrema = [min(y), max(y)]
    width = np.diff(x_extrema)[0]
    height = np.diff(y_extrema)[0]

    radius = np.mean([width, height])/2
    center = [np.mean(x_extrema), np.mean(y_extrema)]

    return (width, height, radius, center)


def grab_paths(session_folder):
    pattern_dict = make_pattern_dict()

    paths = get_data_paths(session_folder, pattern_dict)

    return paths


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return (phi, rho)