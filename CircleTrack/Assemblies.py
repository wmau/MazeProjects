import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import zscore
import os
from tqdm import tqdm
from CaImaging import util

def plot_assembly(pattern, activation, spike_times, sort_spike_times=True, ax=None):
    # Define axes.
    if ax is None:
        fig, ax = plt.subplots()
    activation_ax = ax
    spikes_ax = activation_ax.twinx()

    # Get sort order for neurons based on contribution to assembly.
    if sort_spike_times:
        order = np.argsort(np.abs(pattern))
        sorted_spike_times = [spike_times[n] for n in order]
    else:
        sorted_spike_times = spike_times

    activation_ax.plot(activation)
    activation_ax.set_ylabel('Activation strength [a.u.]')
    activation_ax.set_xlabel('Frame #')
    spikes_ax.eventplot(sorted_spike_times, color='k')
    spikes_ax.set_ylabel('Neurons')

    return activation_ax, spikes_ax

def write_assembly_triggered_movie(activation, frame_numbers, behavior_movie,
                                   fpath=None, threshold=2.58):
    z_activation = zscore(activation)
    above_threshold_frames = frame_numbers[z_activation > threshold]

    grouped_frames = util.cluster(above_threshold_frames, 30)

    compressionCodec = "FFV1"
    codec = cv2.VideoWriter_fourcc(*compressionCodec)
    cap = cv2.VideoCapture(behavior_movie)
    ret, frame = cap.read()
    blank_frame = np.zeros_like(frame)
    rows, cols = int(cap.get(4)), int(cap.get(3))

    if fpath is None:
        folder = os.path.split(behavior_movie)[0]
        fpath = os.path.join(folder, 'Assembly activity.avi')

    writeFile = cv2.VideoWriter(fpath, codec, 15, (cols, rows),
                                isColor=True)
    print(f'Writing {fpath}')
    for group in grouped_frames:
        for frame_number in group:
            cap.set(1, frame_number)
            ret, frame = cap.read()
            writeFile.write(frame)

        for i in range(15):
            writeFile.write(blank_frame)

    writeFile.release()
    cap.release()

