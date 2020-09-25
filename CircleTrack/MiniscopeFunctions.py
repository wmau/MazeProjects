from CircleTrack.BehaviorFunctions import BehaviorSession, spiral_plot
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import sync_data
from util import Session_Metadata
from CircleTrack.BehaviorFunctions import linearize_trajectory
from CaImaging.util import ScrollPlot
from CircleTrack.plotting import plot_spiral

class CalciumSession:
    def __init__(self, session_folder):
        self.folder = session_folder
        self.BehaviorSession = BehaviorSession(self.folder)

        self.data = dict()
        self.minian_path = self.BehaviorSession.paths['minian']
        timestamp_paths = self.BehaviorSession.paths['timestamps']
        self.data['behavior'], \
        self.data['imaging'], _ = \
            sync_data(self.BehaviorSession.behavior_df,
                      self.minian_path,
                      timestamp_paths)


    def plot_spatial_response(self, cell_number=0):
        spikes = [activity > np.std(activity) for activity in
                  self.data['imaging']['S']]
        t = np.asarray(self.data['behavior'].frame)
        lin_position = np.asarray(linearize_trajectory(self.data['behavior'])[0])
        cell_number_labels = [f'Cell #{n}' for n, _ in enumerate(spikes)]
        self.spiral_spatial_plot = ScrollPlot(plot_spiral,
                                              current_position=cell_number,
                                              t = t, lin_position=lin_position,
                                              markers = spikes,
                                              marker_legend = 'Spikes',
                                              subplot_kw={'projection': 'polar'},
                                              titles=cell_number_labels)
        pass
        # fig, ax = spiral_plot(self.data['behavior'],  trace > std,
        #                       marker_legend='spike')
        # ax.set_title(f'Cell # {cell_number}')



if __name__ == '__main__':
    folder = r'Z:\Will\Drift\Data\Castor_Scope05\09_09_2020_CircleTrackGoals2\16_46_11'
    S = CalciumSession(folder)
    S.plot_spatial_response()
    pass