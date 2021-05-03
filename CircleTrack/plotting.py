import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from CaImaging.util import check_attrs


def plot_spiral(ScrollObj):
    """
    Scroll through rows of activation of cells/assemblies. ScrollObj should have the following fields:

    t: array-like
        Timestamps, usually from the t column of behavior_df.

    lin_position: array-like, same size as t
        Linearized position of the mouse.

    markers: (n, t) boolean array-likes
        Array that's True where a cell/assembly is active. The function will scroll through rows (n).

    marker_legend: str
        To indicate where the activations are from.

    lin_ports: array-like or list
        Linearized positions (polar coordinates) of the port locations.

    rewarded: boolean array-like or list
        To indicate whether each port was rewarded.

    :param ScrollObj:
    :return:
    """
    attrs = ["t", "lin_position", "markers", "marker_legend", "lin_ports", "rewarded"]
    check_attrs(ScrollObj, attrs)

    ax = ScrollObj.ax
    lin_position = ScrollObj.lin_position
    t = ScrollObj.t
    lin_ports = ScrollObj.lin_ports
    rewarded = ScrollObj.rewarded
    this_marker_set = ScrollObj.markers[ScrollObj.current_position]

    colors = {True: 'g',
              False: 'r',
              }

    ax.plot(lin_position, t)
    ax.plot(lin_position[this_marker_set], t[this_marker_set], "ro", markersize=2)
    for port, rewarded in zip(lin_ports, rewarded):
        ax.axvline(x=port, color=colors[rewarded])

    ax.legend(["Trajectory", ScrollObj.marker_legend])

    ax.spines["polar"].set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ScrollObj.last_position = ScrollObj.markers.shape[0] - 1


def plot_raster(ScrollObj):
    attrs = ["rasters", "tuning_curves", "binary"]
    check_attrs(ScrollObj, attrs)

    axs = ScrollObj.ax
    rasters = ScrollObj.rasters
    tuning_curve = ScrollObj.tuning_curves
    if ScrollObj.binary:
        cmap = "gray"
    else:
        cmap = "binary"
    axs[0].set_title(ScrollObj.titles[ScrollObj.current_position])
    axs[0].imshow(rasters[ScrollObj.current_position], cmap=cmap)
    axs[1].plot(tuning_curve[ScrollObj.current_position])

    ScrollObj.last_position = len(ScrollObj.rasters)


def plot_daily_rasters(ScrollObj):
    attrs = ["rasters", "tuning_curves"]
    check_attrs(ScrollObj, attrs)

    axs = ScrollObj.ax
    rasters = ScrollObj.rasters
    tuning_curves = ScrollObj.tuning_curves

    for day, (raster, tuning_curve) in enumerate(zip(rasters, tuning_curves)):
        axs[day, 0].imshow(raster[ScrollObj.current_position], cmap='gray', aspect='auto')
        axs[day, 0].set_ylabel('Laps')
        axs[day, 1].plot(tuning_curve[ScrollObj.current_position])
        axs[day, 1].set_ylabel('Transient\nrate')
    axs[day, 0].set_xlabel('Linearized position')
    axs[day, 1].set_xlabel('Linearized position')
    plt.tight_layout()

    ymax = []
    ymin = []
    for ax in axs:
        ylims = ax[1].get_ylim()
        ymin.append(ylims[0])
        ymax.append(ylims[1])

    for ax in axs:
        ax[1].set_ylim([min(ymin), max(ymax)])

    ScrollObj.last_position = rasters[0].shape[0] - 1


def spiral_plot(t, lin_position, markers, ax=None, marker_legend="Licks"):
    """
    Plot trajectory of the mouse over time in a circular (polar) axis. Theta
    corresponds to the animal's position while the radius (distance from center)
    is time. Also plot events of interest (e.g., licks or calcium activity).

    :parameters
    ---
    t: array
        Time vector, usually np.asarray(behavior_df.frame).

    lin_position: array
        Vector of polar coordinates (angles) for each time point t.

    markers: array
        Something that indexes lin_position. These locations will be
        marked as "events" on the spiral plot.

    marker_legend: str
        Label for whatever you are highlighting
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
    ax.plot(lin_position, t)
    ax.plot(lin_position[markers], t[markers], "ro", markersize=2)
    ax.legend(["Trajectory", marker_legend])

    # Clean up axes.
    ax.spines["polar"].set_visible(False)
    ax.set_xticklabels([])
    # ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")  # Make 12 o'clock "0 degrees".
    ax.set_theta_direction(-1)  # Polar coordinates go counterclockwise.
    # ax.set_yticklabels(t_labels)

    return ax

def highlight_column(column, ax, **kwargs):
    ylims = ax.get_ylim()
    rect = plt.Rectangle((column-0.5, ylims[1]), 1, np.abs(np.diff(ylims)), fill=False, **kwargs)
    ax.add_patch(rect)

    return rect

