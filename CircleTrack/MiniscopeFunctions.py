from CircleTrack.BehaviorFunctions import BehaviorSession, spiral_plot
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import sync_data
from util import Session_Metadata

class Session:
    def __init__(self, session_folder):
        self.folder = session_folder
        self.behavior = BehaviorSession(self.folder)

        self.data = dict()
        self.minian_path = self.behavior.paths['minian']
        timestamp_paths = self.behavior.paths['timestamps']
        self.data['behavior'], \
        self.data['imaging'], _ = \
            sync_data(self.behavior.behavior_df,
                      self.minian_path,
                      timestamp_paths)

    def plot_spatial_response(self, cell_number=0):
        trace = self.data['imaging']['S'][cell_number]
        std = np.std(trace)
        fig, ax = spiral_plot(self.data['behavior'],  trace > std,
                              marker_legend='spike')
        ax.set_title(f'Cell # {cell_number}')

        pass

if __name__ == '__main__':
    folder = r'Z:\Will\Drift\Data\Castor_Scope05\09_09_2020_CircleTrackGoals2\16_46_11'
    S = Session(folder)
    pass