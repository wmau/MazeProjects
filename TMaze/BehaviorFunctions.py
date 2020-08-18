import tkinter as tk
tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from CaImaging.util import ScrollPlot, disp_frame, consecutive_dist
from CaImaging.Behavior import read_eztrack, convert_dlc_to_eztrack
from scipy.stats import zscore

metadata_csv = r'Z:\Will\Drift\Data\Metadata.csv'

class Preprocess:
    def __init__(self, metadata_csv=metadata_csv, folder=None):
        """
        Preprocess position data on the TMaze. Almost a clone of the
        circle track Preprocess class, except more advanced. Here, I'm
        pulling metadata (e.g., camera numbers) from a metadata file.
        At some point, I should update the circle track Preprocess
        class to do this the same way.

        :param folder:
        """
        # Get the folder you want to analyze.
        if folder is None:
            self.folder = filedialog.askdirectory()
        else:
            self.folder = folder

        # Read metadata from the master csv.
        metadata_df = pd.read_csv(metadata_csv)

        # Get some metadata.
        entry_match = metadata_df.loc[metadata_df['Path'] == folder]
        self.camera_numbers = {'miniscope': int(entry_match['MiniscopeCam']),
                               'behavior': int(entry_match['BehaviorCam'])}
        keys = ['BehaviorVideo', 'Timestamps', 'BehaviorData']
        self.paths = {key: str(entry_match[key].iloc[-1])
                      for key in keys}
        self.paths['PreprocessedBehavior'] = \
            os.path.join(self.folder, 'PreprocessedBehavior.csv')

        # Try loading previously preprocessed data.
        try:
            self.behavior_df = pd.read_csv(self.paths['PreprocessedBehavior'])

        # Otherwise, do it now.
        except:
            try:
                self.behavior_df = read_eztrack(self.paths['BehaviorData'])
            except:
                convert_dlc_to_eztrack(self.paths['BehaviorData'])
                print('DeepLabCut .h5 successfully converted to .csv.')
                self.behavior_df = read_eztrack(self.paths['BehaviorData'])

            # Get initial positions.
            self.interp_mistracks()


    def save(self, path=None, fname='PreprocessedBehavior.csv'):
        """
        Save preprocessed data.

        path: str
            Folder path to save to. If None, default to session folder.

        fname: str
            File name to call the pkl file.

        """
        if path is None:
            fpath = self.paths['PreprocessedBehavior']
        else:
            fpath = os.path.join(path, fname)

        self.behavior_df.to_csv(fpath, index=False)


    def interp_mistracks(self, thresh=4):
        """
        Z-score the velocity and find abnormally fast movements. Interpolate those.

        :parameter
        ---
        thresh: float
            Number of standard deviations above the mean to be called a mistrack.
        """
        mistracks = zscore(self.behavior_df['distance']) > thresh
        self.behavior_df.loc[mistracks, ['x','y']] = np.nan
        self.behavior_df.interpolate(method='linear', columns=['x', 'y'],
                                     inplace=True)


    def plot_frames(self, frame_number):
        """
        Plot frame and position from ezTrack csv.

        :parameter
        frame_num: int
            Frame number that you want to start on.
        """
        vid = cv2.VideoCapture(self.paths['BehaviorVideo'], cv2.CAP_FFMPEG)
        n_frames = int(vid.get(7))
        frame_nums = ["Frame " + str(n) for n in range(n_frames)]
        self.f = ScrollPlot(disp_frame,
                            current_position=frame_number,
                            vid_fpath=self.paths['BehaviorVideo'],
                            x=self.behavior_df['x'], y=self.behavior_df['y'],
                            titles=frame_nums)


    def correct_position(self, start_frame=None):
        """
        Correct position starting from start_frame. If left to default,
        start from where you specified during class instantiation or where
        you last left off.

        :parameter
        ---
        start_frame: int
            Frame number that you want to start on.
        """
        # Frame to start on.
        if start_frame is None:
            start_frame = 0

        # Plot frame and position, then connect to mouse.
        self.plot_frames(start_frame)
        self.f.fig.canvas.mpl_connect('button_press_event',
                                      self.correct)

        # Wait for click.
        while plt.get_fignums():
            plt.waitforbuttonpress()

        #self.preprocess()


    def correct(self, event):
        """
        Defines what happens during mouse clicks.

        :parameter
        ---
        event: click event
            Defined by mpl_connect. Don't modify.
        """
        # Overwrite DataFrame with new x and y values.
        self.behavior_df.loc[
            self.f.current_position, 'x'] = event.xdata
        self.behavior_df.loc[
            self.f.current_position, 'y'] = event.ydata

        # Plot the new x and y.
        self.f.fig.axes[0].plot(event.xdata, event.ydata, 'go')
        self.f.fig.canvas.draw()


    def find_outliers(self):
        """
        Plot the distances between points and then select with your
        mouse where you would want to do a manual correction from.

        """
        # Plot distance between points and connect to mouse.
        # Clicking the plot will bring you to the frame you want to
        # correct.
        self.traj_fig, self.ax = plt.subplots(1,1,num='outliers')

        # Re-do linearize trajectory and velocity calculation.
        self.behavior_df['distance'][1:] = \
            consecutive_dist(np.asarray((self.behavior_df.x, self.behavior_df.y)).T,
                             axis=0)

        # Plot.
        self.ax.plot(self.behavior_df['distance'], color='r', alpha=0.5)
        self.ax.set_ylabel('Velocity')
        self.ax.set_xlabel('Frame')
        self.traj_fig.canvas.mpl_connect('button_press_event',
                                          self.jump_to)

        while plt.fignum_exists('outliers'):
            plt.waitforbuttonpress()


    def jump_to(self, event):
        """
        Jump to this frame based on a click on a graph. Grabs the x (frame)
        """
        plt.close(self.traj_fig)
        if event.xdata < 0:
            event.xdata = 0
        self.correct_position(int(np.round(event.xdata)))


if __name__ == '__main__':
    folder = r'Z:\Will\Drift\Data\M1\07_12_2020_TMazeFreeChoice2\H15_M23_S4'
    P = Preprocess(folder=folder)

    pass