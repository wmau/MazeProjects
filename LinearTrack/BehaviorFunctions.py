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
)
def find_water_ports_LT(behavior_df):
    ports = {'x': np.zeros(2),
             'y': np.zeros(2),
             }
    ports = pd.DataFrame(ports)

    for port in range(2):
        licking = behavior_df["lick_port"] == port

        x = np.median(behavior_df.loc[licking, "x"])
        y = np.median(behavior_df.loc[licking, "y"])

        ports.loc[port, "x"] = x
        ports.loc[port, "y"] = y

    return ports

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

            self.behavior_df = clean_lick_detection(self.behavior_df,
                                                    linear_track=True)
            self.preprocess()

    def preprocess(self):
        """
        Fill in DataFrame columns with some calculated values:
            Linearized position
            Trial number
            Distance (velocity)

        """
        self.behavior_df["trials"], self.behavior_df['direction'] = \
            get_trials(self.behavior_df.x)
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

    def quick_manual_correct(self, velocity_threshold=15):
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

def get_trials(x, nbins=16):
    # Bin position.
    bins = np.linspace(min(x), max(x), nbins)
    binned_position = np.digitize(x, bins)
    bins = np.unique(binned_position)

    # For each bin number, get timestamps when the mouse was in that bin.
    indices = [np.where(binned_position == this_bin)[0] for this_bin in bins]

    # Preallocate the trial array.
    trials = np.full(binned_position.shape, np.nan)
    direction = np.full(binned_position.shape, np.nan, dtype=object)

    trial_start = 0     # Trial starts on first frame.

    # Where is the mouse placed on the track? If closer to the left side,
    # the first trial ends when the mouse enters the rightmost bin.
    start_i = {'left': 0,
               'right': -2}
    stop_i = {'left': -2,
              'right': 0}
    if bins[0] < nbins/2:
        start_side = 'left'
    else:
        start_side = 'right'

    # For a larg enumber of trials...
    for trial_number in range(500):
        # Find the frames where the mouse is in a certain bin.
        start_bin_frames = indices[start_i[start_side]]
        stop_bin_frames = indices[stop_i[start_side]]

        # Find when the mouse enters the start bin and the stop bin.
        enter_start = start_bin_frames[np.argmax(start_bin_frames > trial_start)]
        trial_end = stop_bin_frames[np.argmax(stop_bin_frames > enter_start)]

        # Switch sides.
        if start_side == 'right':
            start_side = 'left'
        else:
            start_side = 'right'

        # Assign the trial number.
        if np.all(np.isnan(trials[trial_start:trial_end])):
            trials[trial_start:trial_end] = trial_number
            direction[trial_start:trial_end] = start_side
        else:
            trials[np.isnan(trials)] = trial_number - 1
            break

        # Start the next trial when the last trial ends.
        trial_start = trial_end

    return trials.astype(int), direction

if __name__ == '__main__':
    P = Preprocess(
        r'Z:\Will\LinearTrack\Data\Atlas\2021_05_24_LinearTrack1\09_51_14')
    #P.find_outliers()

    pass