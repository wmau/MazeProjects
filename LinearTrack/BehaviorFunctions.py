from tkinter import filedialog
from util import Session_Metadata
from CaImaging.Behavior import convert_dlc_to_eztrack
import pandas as pd
import os
import numpy as np
from util import find_timestamp_file
import cv2
from CaImaging.util import (
    ScrollPlot,
    disp_frame,
    consecutive_dist,
)
import matplotlib.pyplot as plt
from scipy.stats import zscore
from CircleTrack.BehaviorFunctions import (
    sync_Arduino_outputs,
    clean_lick_detection,
    make_tracking_video,
    get_trials
)

class Preprocess:
    def __init__(
        self, folder=None, behav_cam=1, miniscope_cam=0, subtract_offset=False
    ):
        """
        Preprocesses behavior data by specifying a session folder.

        :parameter
        ---
        folder: str
            Folder path to session.
        """
        if folder is None:
            self.folder = filedialog.askdirectory()
        else:
            self.folder = folder

        # Get the paths to relevant files.
        self.paths = Session_Metadata(self.folder).meta_dict
        self.paths["PreprocessedBehavior"] = os.path.join(
            self.folder, "PreprocessedBehavior.csv"
        )

        if type(self.paths["timestamps"]) is str:
            self.camera_numbers = {
                "behav_cam": behav_cam,
                "miniscope_cam": miniscope_cam,
            }
            self.v4 = False
        else:
            self.v4 = True

        # Check if Preprocess has been ran already by attempting
        # to load a pkl file.
        try:
            self.behavior_df = pd.read_csv(self.paths["PreprocessedBehavior"])

        # If not, sync Arduino data.
        except:
            if not self.paths["BehaviorData"]:
                try:
                    convert_dlc_to_eztrack(self.paths["DLC"])
                    print("DeepLabCut .h5 successfully converted to .csv.")
                    self.paths = Session_Metadata(self.folder, overwrite=True).meta_dict
                    self.paths["PreprocessedBehavior"] = os.path.join(
                        self.folder, "PreprocessedBehavior.csv"
                    )

                except:
                    print("DLC file not detected. Reading ezTrack file instead.")

            self.behavior_df = sync_Arduino_outputs(
                self.folder,
                behav_cam=behav_cam,
                miniscope_cam=miniscope_cam,
                subtract_offset=subtract_offset,
            )[0]

            # Find timestamps where the mouse seemingly teleports to a new location.
            # This is likely from mistracking. Interpolate those data points.
            self.interp_mistracks()

            self.behavior_df = clean_lick_detection(self.behavior_df)
            self.preprocess()

    def preprocess(self):
        """
        Fill in DataFrame columns with some calculated values:
            Linearized position
            Trial number
            Distance (velocity)

        """
        self.behavior_df["trials"] = get_trials(self.behavior_df, do_linearize=False, circular_bin=False)
        self.behavior_df["distance"] = consecutive_dist(
            np.asarray((self.behavior_df.x, self.behavior_df.y)).T, zero_pad=True
        )
        self.behavior_df["t"] = self.get_timestamps()

    def save(self, path=None, fname="PreprocessedBehavior.csv"):
        """
        Save preprocessed data.

        path: str
            Folder path to save to. If None, default to session folder.

        fname: str
            File name to call the pkl file.

        """
        if path is None:
            fpath = self.paths["PreprocessedBehavior"]
        else:
            fpath = os.path.join(path, fname)

        self.behavior_df.to_csv(fpath, index=False)

    def auto_find_outliers(self, velocity_threshold=40):
        jump_frames = np.where((self.behavior_df["distance"] > velocity_threshold))[0]
        while any(jump_frames):
            self.correct_position(jump_frames[0])

            jump_frames = np.where(
                (self.behavior_df["distance"] > velocity_threshold))[
                0]

    def get_timestamps(self):
        if not self.v4:
            timestamps = pd.read_csv(self.paths["timestamps"], sep="\t")
            timestamps = timestamps.loc[
                timestamps["camNum"] == self.camera_numbers["behav_cam"]
            ]
            t = timestamps["sysClock"]
        else:
            timestamp_file = find_timestamp_file(self.paths["timestamps"], "BehavCam")
            timestamps = pd.read_csv(timestamp_file)
            t = timestamps["Time Stamp (ms)"]

        t.reset_index(inplace=True, drop=True)
        t.iloc[0] = 0

        return t

    def interp_mistracks(self, thresh=4):
        """
        Z-score the velocity and find abnormally fast movements. Interpolate those.

        :parameter
        ---
        thresh: float
            Number of standard deviations above the mean to be called a mistrack.
        """
        mistracks = zscore(self.behavior_df["distance"]) > thresh
        self.behavior_df.loc[mistracks, ["x", "y"]] = np.nan
        self.behavior_df.interpolate(method="linear", columns=["x", "y"], inplace=True)

    def plot_frames(self, frame_number):
        """
        Plot frame and position from ezTrack csv.

        :parameter
        frame_num: int
            Frame number that you want to start on.
        """
        vid = cv2.VideoCapture(self.paths["BehaviorVideo"], cv2.CAP_FFMPEG)
        n_frames = int(vid.get(7))
        frame_nums = ["Frame " + str(n) for n in range(n_frames)]
        self.f = ScrollPlot(
            disp_frame,
            current_position=frame_number,
            vid_fpath=self.paths["BehaviorVideo"],
            x=self.behavior_df["x"],
            y=self.behavior_df["y"],
            titles=frame_nums,
        )

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
        self.f.fig.canvas.mpl_connect("button_press_event", self.correct)

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
        self.behavior_df.loc[self.f.current_position, "x"] = event.xdata
        self.behavior_df.loc[self.f.current_position, "y"] = event.ydata

        # Plot the new x and y.
        self.f.fig.axes[0].plot(event.xdata, event.ydata, "go")
        self.f.fig.canvas.draw()

    def find_outliers(self):
        """
        Plot the distances between points and then select with your
        mouse where you would want to do a manual correction from.

        """
        # Plot distance between points and connect to mouse.
        # Clicking the plot will bring you to the frame you want to
        # correct.
        self.traj_fig, self.traj_ax = plt.subplots(1, 1, num="outliers")
        self.dist_ax = self.traj_ax.twinx()

        # Re-do linearize trajectory and velocity calculation.
        self.behavior_df["distance"][1:] = consecutive_dist(
            np.asarray((self.behavior_df.x, self.behavior_df.y)).T, axis=0
        )

        # Plot.
        self.traj_ax.plot(self.behavior_df['x'], alpha=0.5)
        self.traj_ax.set_ylabel("Distance from center", color="b")
        self.dist_ax.plot(self.behavior_df["distance"], color="r", alpha=0.5)
        self.dist_ax.set_ylabel("Velocity", color="r", rotation=-90)
        self.dist_ax.set_xlabel("Frame")
        self.traj_fig.canvas.mpl_connect("button_press_event", self.jump_to)

        try:
            while plt.fignum_exists("outliers"):
                plt.waitforbuttonpress()
        except KeyboardInterrupt:
            self.save()

    def jump_to(self, event):
        """
        Jump to this frame based on a click on a graph. Grabs the x (frame)
        """
        plt.close(self.traj_fig)
        if event.xdata < 0:
            event.xdata = 0
        self.correct_position(int(np.round(event.xdata)))

    def plot_lin_position(self):
        """
        Plots the linearized position for the whole session, color-coded by trial.

        """
        for trial in range(int(max(self.behavior_df["trials"]))):
            plt.plot(
                self.behavior_df["lin_position"][self.behavior_df["trials"] == trial]
            )

    def plot_trial(self, trial):
        """
        Plots any trial (non-linearized).

        :parameter
        ---
        trial: int
            Trial number
        """
        x = self.behavior_df["x"]
        y = self.behavior_df["y"]
        idx = self.behavior_df["trials"] == trial

        fig, ax = plt.subplots()
        ax.plot(x[idx], y[idx])
        ax.set_aspect("equal")

    def track_video(self, start=0, stop=None, fname="Tracking.avi", fps=15):
        make_tracking_video(
            self.folder, start=start, stop=stop, output_fname=fname, fps=fps
        )

if __name__ == '__main__':
    P = Preprocess(
        r'Z:\Will\LinearTrack\Data\Miranda'
        r'\2021_03_08_LinearTrackShaping1\10_04_48')
    P.find_outliers()

    pass