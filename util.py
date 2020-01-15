from util import get_data_paths
import os

def make_pattern_dict():
    pattern_dict = {}
    pattern_dict['Arduino'] = 'H??_M??_S??.???? ????.txt'
    pattern_dict['MiniscopeFolder'] = 'H??_M??_S??'
    pattern_dict['BehaviorVideo'] = os.path.join(pattern_dict['MiniscopeFolder'],
                                                 'Merged.avi')
    pattern_dict['EZTrack'] = os.path.join(pattern_dict['MiniscopeFolder'],
                                           '*_LocationOutput.csv')
    pattern_dict['settings'] = os.path.join(pattern_dict['MiniscopeFolder'],
                                            'settings_and_notes.dat')
    pattern_dict['timestamps'] = os.path.join(pattern_dict['MiniscopeFolder'],
                                              'timestamp.dat')

    return pattern_dict