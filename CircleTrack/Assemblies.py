import numpy as np
import matplotlib.pyplot as plt

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
    spikes_ax.eventplot(sorted_spike_times, color='k')

