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
    find_water_ports,
    find_rewarded_ports,
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

            # self.behavior_df = clean_lick_detection(self.behavior_df,
            #                                         linear_track=True)
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

    def final_save(self):
        self.behavior_df = clean_lick_detection(self.behavior_df,
                                                linear_track=True)

        self.save()

    def quick_manual_correct(self, velocity_threshold=25):
        jump_frames = np.where((self.behavior_df["distance"] > velocity_threshold))[0]
        while any(jump_frames):
            self.correct_position(jump_frames[0])

            jump_frames = np.where(
                (self.behavior_df["distance"] > velocity_threshold))[
                0]

            self.save()

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

        self.preprocess()

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
            direction[np.isnan(trials)] = start_side
            break

        # Start the next trial when the last trial ends.
        trial_start = trial_end

    return trials.astype(int), direction

class BehaviorSession:
    def __init__(self, folder=None, pix_per_cm=6.3):
        """
        Contains many useful analyses for single session data.

        :parameter
        ---
        folder: str
            Directory pertaining to a single session. If not specified, a
            dialog box will appear for you to navigate to the folder.

        pix_per_cm: float
            Number of pixels per cm. 6.21 for the newest ROI, 6.56 for the older.

        Useful methods:
            plot_licks(): Plots the trajectory and licks in a spiral pattern.

            port_approaches(): Plots speed or acceleration within a time
                window centered around the arrival to each port.

            sdt_trials(): Gets the hit/miss/false alarm/correct rejection rate
                for each port. Currently in beta.

        """
        # If folder is not specified, open a dialog box.
        self.meta = dict()
        if folder is None:
            self.meta["folder"] = filedialog.askdirectory()
        else:
            self.meta["folder"] = folder

        # Find paths.
        self.meta["paths"] = Session_Metadata(self.meta["folder"]).meta_dict
        self.meta["paths"]["PreprocessedBehavior"] = os.path.join(
            self.meta["folder"], "PreprocessedBehavior.csv"
        )

        # Determine if this was a recording with the new QT system.
        self.meta["v4"] = (
            True if type(self.meta["paths"]["timestamps"]) is list else False
        )

        # Try loading a presaved csv.
        self.data = dict()
        try:
            self.data["df"] = pd.read_csv(self.meta["paths"]["PreprocessedBehavior"])
        except:
            raise FileNotFoundError("Run Preprocess() first.")

        # Get timing information.
        if self.meta["v4"]:
            self.meta["behavior_timestamps"] = pd.read_csv(
                find_timestamp_file(self.meta["paths"]["timestamps"], "BehavCam")
            )
        else:
            self.meta["behavior_timestamps"] = pd.read_csv(
                self.meta["paths"]["timestamps"], sep="\t"
            )
        self.meta['local'] = False

        self.meta["fps"] = self.get_fps()

        # Convert x, y, and distance values to cm.
        self.data["df"]["x"] = self.data["df"]["x"] / pix_per_cm
        self.data["df"]["y"] = self.data["df"]["y"] / pix_per_cm
        self.data["df"]["distance"] = self.data["df"]["distance"] / pix_per_cm

        # Number of laps run.
        self.data["ntrials"] = max(self.data["df"]["trials"] + 1)

        # Amount of time spent per trial (in frames).
        self.data["frames_per_trial"] = np.bincount(self.data["df"]["trials"])

        # Find water ports.
        self.data["ports"], self.data["lin_ports"] = find_water_ports(self.data["df"],
                                                                      linear_track=True,
                                                                      use_licks=True)
        self.data["n_drinks"] = self.count_drinks()


    def get_fps(self):
        """
        Get sampling frequency of behavior video. We don't trust the
        cv2 method for extracting fps from video files. Instead,
        count time intervals in between video frames in timeStamp.csv.

        :return:
        """
        # Take difference.
        if self.meta["v4"]:
            key = "Time Stamp (ms)"
        else:
            key = "sysClock"

        interframe_intervals = np.diff(self.meta["behavior_timestamps"][key].iloc[1:])

        # Inter-frame interval in milliseconds.
        mean_interval = np.mean(interframe_intervals)
        fps = round(1 / (mean_interval / 1000))

        return fps

    def count_drinks(self):
        """
        Count number of rewards retrieved per trial.

        :return:
        """
        n_rewards = []
        for trial in range(self.data["ntrials"]):
            water_deliveries = np.sum(
                self.data["df"].loc[self.data["df"]["trials"] == trial]["water"]
            )
            n_rewards.append(water_deliveries)

        return np.asarray(n_rewards)

    def plot_licks(self):
        """
        Plots licks as a line plot.

        :return:
        """
        fig, ax = plt.subplots()
        ax.plot(
            self.data["all_licks"][:, self.data["rewarded_ports"]], "cornflowerblue"
        )
        ax.plot(
            self.data["all_licks"][:, ~self.data["rewarded_ports"]], "gray", alpha=0.6
        )
        ax.set_xlabel("Trials")
        ax.set_ylabel("Licks")


if __name__ == '__main__':
    P = Preprocess(
        r'Z:\Will\LinearTrack\Data\Miranda\2021_03_12_LinearTrack3\09_58_05')
    #P.find_outliers()

    pass