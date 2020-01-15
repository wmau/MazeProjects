from util import get_data_paths
import os

def make_pattern_dict():
    miniscope_folder = 'H??_M??_S??'
    pattern_dict = {
        'Arduino': 'H??_M??_S??.???? ????.txt',
        'MiniscopeFolder': miniscope_folder,
        'BehaviorVideo': os.path.join(miniscope_folder, 'Merged.avi'),
        'EZTrack': os.path.join(miniscope_folder, '*_LocationOutput.csv'),
        'settings': os.path.join(miniscope_folder, 'settings_and_notes.dat'),
        'timestamps': os.path.join(miniscope_folder, 'timestamp.dat')
    }

    return pattern_dict