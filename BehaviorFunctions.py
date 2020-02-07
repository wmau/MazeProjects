import os
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from LickArduino import clean_Arduino_output
from util import read_eztrack, find_closest, ScrollPlot, disp_frame, \
    consecutive_dist
import matplotlib.pyplot as plt
import cv2
from CircleTrack.utils import circle_sizes, cart2pol, grab_paths
import pickle as pkl
import tkinter as tk
tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog

def make_tracking_video(vid_path, preprocessed=True, csv_path=None,
                        Arduino_path=None, output_fname='Tracking.avi',
                        start=0, stop=None, fps=30):
    """
    Makes a video to visualize licking at water ports and position of the animal.

    :parameters
    ---
    video_fname: str
        Full path to the behavior video.

    csv_fname: str
        Full path to the csv file containing position (x and y).

    output_fname: str
        Desired file name for output. It will be saved to the same folder
        as the data.

    start: int
        Frame to start on.

    stop: int or None
        Frame to stop on or if None, the end of the movie.

    fps: int
        Sampling rate of the behavior camera.

    Arduino_path:
        Full path to the Arduino output txt. If None, doesn't plot licking.

    """
    # Get behavior video.
    vid = cv2.VideoCapture(vid_path)
    if stop is None:
        stop = int(vid.get(7))  # 7 is the index for total frames.

    # Save data to the same folder.
    folder = os.path.split(vid_path)[0]
    output_path = os.path.join(folder, output_fname)

    # Get EZtrack data.
    if preprocessed:
        session_folder = os.path.split(vid_path)[0]
        behav = Preprocess(session_folder)
        eztrack = behav.eztrack_data
    else:
        if Arduino_path is not None:
            eztrack = sync_Arduino_outputs(Arduino_path, csv_path)[0]
            eztrack = clean_lick_detection(eztrack)
        else:
            eztrack = read_eztrack(csv_path)

    # Define the colors that the cursor will flash for licking each port.
    port_colors = ['saddlebrown',
                   'red',
                   'orange',
                   'yellow',
                   'green',
                   'blue',
                   'darkviolet',
                   'gray']

    # Make video.
    fig, ax = plt.subplots()
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, 100):
        for frame_number in np.arange(start, stop):
            # Plot frame.
            vid.set(1, frame_number)
            ret, frame = vid.read()
            ax.imshow(frame)

            # Plot position.
            x = eztrack.at[frame_number, 'x']
            y = eztrack.at[frame_number, 'y']
            ax.scatter(x, y, marker='+', s=60, c='w')

            ax.text(0, 0, 'Frame: ' + str(frame_number) +
                    '   Time: ' + str(np.round(frame_number/30, 1)) + ' s')

            # Lick indicator.
            if (Arduino_path is not None) or preprocessed:
                licking_at_port = eztrack.at[frame_number, 'lick_port']
                if licking_at_port >= 0:
                    ax.scatter(x, y, s=200, marker='+',
                               c=port_colors[licking_at_port])

            ax.set_aspect('equal')
            plt.axis('off')

            writer.grab_frame()

            plt.cla()


def sync_Arduino_outputs(Arduino_fpath, eztrack_fpath, behav_cam=2):
    """
    This function is meant to be used in conjunction with the above
    functions and Miniscope software recordings. Miniscope software
    will save videos (behavior and imaging) in a timestamped folder.

    :param folder:
    :return:
    """
    # Read the txt file generated by Arduino.
    Arduino_data, offset = clean_Arduino_output(Arduino_fpath)
    Arduino_data['Timestamp'] -= offset

    # Read the position data generated by ezTrack.
    eztrack_data = read_eztrack(eztrack_fpath)
    eztrack_data['water'] = False
    eztrack_data['lick_port'] = int(-1)

    # Get timestamping information from DAQ output.
    folder = os.path.split(eztrack_fpath)[0]
    timestamp_fpath = os.path.join(folder, 'timestamp.dat')
    try:
        DAQ_data = pd.read_csv(timestamp_fpath, sep="\s+")
    except:
        raise Exception('DAQ timestamp.dat not found or corrupted.')

    # Only take the rows corresponding to the behavior camera.
    DAQ_data = DAQ_data[DAQ_data.camNum == behav_cam]
    DAQ_data.reset_index(drop=True, inplace=True)

    # Discard data after Miniscope acquisition has stopped.
    sysClock = np.asarray(DAQ_data.sysClock)
    Arduino_data.drop(Arduino_data.index[Arduino_data.Timestamp > sysClock[-1]],
                      inplace=True)

    # Find the frame number associated with the timestamp of a lick.
    for i, row in Arduino_data.iterrows():
        closest_time = find_closest(sysClock, row.Timestamp, sorted=True)[1]
        frame_num = DAQ_data.loc[DAQ_data.sysClock == closest_time]['frameNum'].values[0]

        val = row.Data
        if val.isnumeric():
            eztrack_data.at[frame_num, 'lick_port'] = val
        elif val == 'Water':
            eztrack_data.at[frame_num, 'water'] = True


    eztrack_data = eztrack_data.astype({'frame': int,
                                        'water': bool,
                                        'lick_port': int})
    return eztrack_data, Arduino_data


def find_water_ports(eztrack_data):
    """
    Use the x and y extrema to locate water port locations. Requires that the
    maze be positioned so that a port is at the 12 o'clock position. Which port
    is not important -- the code can be modified for any orientation.

    :parameter
    ---
    eztrack_data: cleaned DataFrame from sync_Arduino_outputs()

    :return
    ---
    ports: DataFrame
        DataFrame with 'x' and 'y' columns corresponding to x and y positions of
        each water port.
    """
    (width, height, radius, center) = circle_sizes(eztrack_data.x, eztrack_data.y)
    theta = np.pi/4     # Angle in between each water port.

    # Determines orientation of the water ports.
    # List the port number of the port at 12 o'clock and count up.
    orientation = [7, 0, 1, 2, 3, 4, 5, 6]
    port_angles = [o * theta for o in orientation]

    # Calculate port locations.
    ports = {}
    ports['x'] = radius * np.cos(port_angles) + center[0]
    ports['y'] = radius * np.sin(port_angles) + center[1]
    ports = pd.DataFrame(ports)

    # Debugging purposes.
    # port_colors = ['saddlebrown',
    #                'red',
    #                'orange',
    #                'yellow',
    #                'green',
    #                'blue',
    #                'darkviolet',
    #                'gray']
    # plt.plot(eztrack_data.x, eztrack_data.y)
    # plt.scatter(center[0], center[1], c='r')
    # for color, (i, port) in zip(port_colors, ports.iterrows()):
    #     plt.scatter(port['x'], port['y'], c=color)
    # plt.axis('equal')

    return ports


def clean_lick_detection(eztrack_data, threshold=80):
    """
    Clean lick detection data by checking that the mouse is near the port during
    a detected lick.

    :parameters
    ---
    eztrack_data: cleaned DataFrame from sync_Arduino_outputs()

    threshold: float
        Distance threshold (in pixels) to be considered "near" the port.

    :return
    ---
    eztrack_data: cleaned DataFrame after eliminating false positives.
    """
    ports = find_water_ports(eztrack_data)

    lick_frames = eztrack_data[eztrack_data.lick_port > -1]
    for i, frame in lick_frames.iterrows():
        frame = frame.copy()
        port_num =  frame.lick_port
        frame_num = frame.frame

        distance = np.sqrt((frame.x - ports.at[port_num, 'x'])**2 +
                           (frame.y - ports.at[port_num, 'y'])**2)

        if distance > threshold:
            eztrack_data.at[frame_num, 'lick_port'] = -1

    return eztrack_data


def linearize_trajectory(eztrack_data, x=None, y=None):
    """
    Linearizes circular track trajectory.

    :parameter
    ---
    eztrack_data: output from read_eztrack()

    :returns
    ---
    angles: array
        Basically the linearized trajectory. Technically it is the
        polar coordinate with the center of the maze as the origin.

    radii: array
        Vector length of polar coordinate. Basically the distance from
        the center. Maybe useful for something.
    """
    # Get circle size.
    if x is None:
        x = eztrack_data.x
    if y is None:
        y = eztrack_data.y
    (width, height, radius, center) = circle_sizes(eztrack_data.x,
                                                   eztrack_data.y)

    # Convert to polar coordinates.
    angles, radii = cart2pol(x-center[0], y-center[1])

    # Shift everything so that 12 o'clock (pi/2) is 0.
    angles += np.pi/2
    angles = np.mod(angles, 2*np.pi)

    return angles, radii


def plot_licks(eztrack_data):
    """
    Plot points where mouse licks.

    :parameter
    ---
    eztrack_data: output from Preprocess

    :return
    ---
    fig, ax: Figure and Axes
        Contains the plots.
    """
    # Make sure licks have been retrieved.
    if 'lick_port' not in eztrack_data:
        raise KeyError('Run sync_Arduino_outputs and clean_lick_detection first.')
    else:
        licks = eztrack_data.lick_port
        licks[licks == -1] = np.nan

    # Linearize mouse's trajectory.
    lin_dist = linearize_trajectory(eztrack_data)[0]

    # Find the water ports and get their linearized location.
    ports = find_water_ports(eztrack_data)
    lin_ports = linearize_trajectory(eztrack_data, ports['x'], ports['y'])[0]

    # Make the array for plotting.
    licks = [lin_ports[port_id] if not np.isnan(port_id) else np.nan for port_id in licks]

    # Plot.
    fig, ax = plt.subplots()
    ax.plot(lin_dist)
    ax.plot(licks, marker='x', markersize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Linearized distance (radians)')

    return fig, ax


def find_rewarded_ports(eztrack_data):
    """
    Find which port numbers are rewarded by looking at the flag
    one timestamp before water delivery. Note that the mouse must
    lick at each rewarded port a handful of times for this to work.

    :parameter
    ---
    eztrack_data: output from Preprocess()

    :return
    ---
    ports: array
        Port numbers that were rewarded.
    """
    if 'water' not in eztrack_data:
        raise KeyError('Run sync_Arduino outputs and clean_lick_detection first.')

    # Get index one before water delivery (the lick that triggered it).
    one_before = np.where(eztrack_data.water)

    # Find unique port numbers.
    rewarded_ports = np.unique(eztrack_data.loc[one_before, 'lick_port'])

    return rewarded_ports[rewarded_ports > -1]


def bin_position(linearized_position):
    """
    Bin radial position.

    :parameter
    ---
    linearized_position: array
        Linearized position (position in radians after passing through linearize_trajectory())

    :return
    ---
    binned: array
        Binned position.
    """
    bins = np.linspace(0, 2*np.pi, 9)
    binned = np.digitize(linearized_position, bins)

    return binned


def get_trials(eztrack_data, counterclockwise=False):
    """
    Labels timestamps with trial numbers. Looks through position indices as the mouse
    passes through bins in a clockwise fashion (default).

    :parameters
    ---
    eztrack_data: output from Preprocess()

    counterclockwise: boolean
        Flag for whether session was run with the mouse running counterclockwise.

    :return
    trials: array, same size as eztrack_data position
        Labels for each timestamp for which trial the mouse is on.
    """
    # Linearize then bin position into one of 8 bins.
    position = linearize_trajectory(eztrack_data)[0]
    binned_position = bin_position(position)
    bins = np.unique(binned_position)

    # For each bin number, get timestamps when the mouse was in that bin.
    indices = [np.where(binned_position == this_bin)[0] for this_bin in bins]
    if counterclockwise: # reverse the order of the bins.
        indices = indices[::-1]

    # Preallocate trial vector.
    trials = np.full(binned_position.shape, np.nan)
    trial_start = 0
    last_idx = 0

    # We need to loop through bins rather than simply looking for border crossings
    # because a mouse can backtrack, which we wouldn't want to count.
    # For a large number of trials...
    for trial_number in range(500):

        # For each bin...
        for this_bin in indices:
            # Find the first timestamp that comes after the first timestamp in the last bin
            # for that trial. Argmax is supposedly faster than np.where.
            last_idx = this_bin[np.argmax(this_bin > last_idx)]

        # After looping through all the bins, remember the last timestamp where there
        # was a bin transition.
        trial_end = last_idx

        # If the slice still has all NaNs, label it with the trial number.
        if np.all(np.isnan(trials[trial_start:trial_end])):
            trials[trial_start:trial_end] = trial_number

        # If not, finis up and exit the loop.
        else:
            trials[np.isnan(trials)] = trial_number
            break

        # The start of the next trial is the end of the last.
        trial_start = trial_end

    # Debugging purposes.
    # for trial in range(int(max(trials))):
    #     plt.plot(position[trials == trial])

    return trials.astype(int)


class Preprocess:
    def __init__(self, folder=None):
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
        self.paths = grab_paths(self.folder)
        self.paths['PreprocessedBehavior'] = \
            os.path.join(self.folder, 'PreprocessedBehavior.csv')

        # Check if Preprocess has been ran already by attempting
        # to load a pkl file.
        try:
            self.eztrack_data  = pd.read_csv(self.paths['PreprocessedBehavior'])

        # If not, sync Arduino data.
        except:
            self.eztrack_data = sync_Arduino_outputs(self.paths['Arduino'],
                                                     self.paths['BehaviorData'])[0]
            self.eztrack_data = clean_lick_detection(self.eztrack_data)
            self.eztrack_data['trials'] = get_trials(self.eztrack_data)
            self.eztrack_data['lin_position'] = linearize_trajectory(self.eztrack_data)[0]


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

        self.eztrack_data.to_csv(fpath)


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
                            x=self.eztrack_data['x'], y=self.eztrack_data['y'],
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


    def correct(self, event):
        """
        Defines what happens during mouse clicks.

        :parameter
        ---
        event: click event
            Defined by mpl_connect. Don't modify.
        """
        # Overwrite DataFrame with new x and y values.
        self.eztrack_data.loc[self.f.current_position, 'x'] = event.xdata
        self.eztrack_data.loc[self.f.current_position, 'y'] = event.ydata

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
        self.traj_fig, self.traj_ax = plt.subplots(1,1)
        self.dist_ax = self.traj_ax.twinx()

        angles, radii = linearize_trajectory(self.eztrack_data)
        self.eztrack_data['distance'][1:] = \
            consecutive_dist(np.asarray((self.eztrack_data.x, self.eztrack_data.y)).T,
                             axis=0)

        self.traj_ax.plot(radii)
        self.traj_ax.set_ylabel('Distance from center')
        self.dist_ax.plot(self.eztrack_data['distance'], color='r')
        self.dist_ax.set_ylabel('Velocity')
        self.dist_ax.set_xlabel('Frame')
        self.traj_fig.canvas.mpl_connect('button_press_event',
                                          self.jump_to)

        while plt.get_fignums():
            plt.waitforbuttonpress()


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
        for trial in range(int(max(self.eztrack_data['trials']))):
            plt.plot(self.eztrack_data['lin_position'][self.eztrack_data['trials'] == trial])


    def plot_trial(self, trial):
        """
        Plots any trial (non-linearized).

        :parameter
        ---
        trial: int
            Trial number
        """
        x = self.eztrack_data['x']
        y = self.eztrack_data['y']
        idx = self.eztrack_data['trials'] == trial

        fig, ax = plt.subplots()
        ax.plot(x[idx], y[idx])
        ax.set_aspect('equal')


    def track_video(self):
        make_tracking_video(self.paths['BehaviorVideo'], self.paths['BehaviorData'],
                            Arduino_path=self.paths['Arduino'])


class Process:
    def __init__(self, folder=None):
        # If folder is not specified, open a dialog box.
        if folder is None:
            self.folder = filedialog.askdirectory()
        else:
            self.folder = folder

        self.paths = grab_paths(self.folder)
        self.paths['PreprocessedBehavior'] = \
            os.path.join(self.folder, 'PreprocessedBehavior.csv')

        self.data = pd.read_csv(self.paths['PreprocessedBehavior'])



if __name__ == '__main__':
    folder = r'D:\Projects\CircleTrack\Mouse4\01_27_2020\H13_M31_S49'
    #folder = r'D:\Projects\CircleTrack\Mouse1\12_20_2019\H14_M59_S12'
    behav = Process(folder)
    #make_tracking_video(os.path.join(folder, 'Merged.avi'))

    pass