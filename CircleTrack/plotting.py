import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

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
              False: 'gray',
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
    attrs = ["rasters", "tuning_curves", "cmap", "port_bins",
             "rewarded", "interpolation"]
    check_attrs(ScrollObj, attrs)

    axs = ScrollObj.ax
    rasters = ScrollObj.rasters
    tuning_curve = ScrollObj.tuning_curves
    cmap = ScrollObj.cmap
    interpolation = ScrollObj.interpolation

    axs[0].set_title(ScrollObj.titles[ScrollObj.current_position])
    axs[0].imshow(rasters[ScrollObj.current_position], cmap=cmap, interpolation=interpolation)
    axs[1].plot(tuning_curve[ScrollObj.current_position])
    axs[0].set_ylabel('Laps')
    axs[1].set_xlabel('Linearized position')
    axs[1].set_ylabel('Average transient rate')
    [ax.axis('auto') for ax in axs]

    port_colors = {True: 'g',
                   False: 'gray',
                   }
    alphas = [0.6 if rewarded else 0.2 for rewarded in ScrollObj.rewarded]
    for ax in axs:
        for port, rewarded, alpha in zip(ScrollObj.port_bins, ScrollObj.rewarded, alphas):
            ax.axvline(x=port, color=port_colors[rewarded], alpha=alpha)

    ScrollObj.last_position = len(ScrollObj.rasters)


def plot_port_activations(ScrollObj):
    attrs = ['port_activations', 't_xaxis', 'n_lick_laps', 'rewarded', 'titles']
    check_attrs(ScrollObj, attrs)

    axs = ScrollObj.ax
    port_activations = ScrollObj.port_activations
    t_xaxis = ScrollObj.t_xaxis
    n_lick_laps = ScrollObj.n_lick_laps
    rewarded = ScrollObj.rewarded
    fig_titles = ScrollObj.titles
    fig = ScrollObj.fig
    try:
        previously_rewarded = ScrollObj.previously_rewarded
    except:
        previously_rewarded = []

    for port, (ax, port_activation, lick_approach_sep) in enumerate(
            zip(axs.flatten(),
                port_activations[ScrollObj.current_position],
                n_lick_laps)
    ):
        ax.imshow(port_activation,
                  extent=[t_xaxis[0], t_xaxis[-1], len(port_activation)+1, 1],
                  aspect='auto',
                  cmap='Blues'
                  )
        ax.axhline(y=lick_approach_sep, color='k')
        ax.axvline(x=0, color='r')

        if port in rewarded:
            title_color = 'g'
        elif port in previously_rewarded:
            title_color = 'orange'
        else:
            title_color = 'k'
        ax.set_title(f'Port # {port}', color=title_color)

    max_clim = np.max([np.nanmax(activation) for activation in
                       port_activations[ScrollObj.current_position]])
    for ax in fig.axes:
        for im in ax.get_images():
            im.set_clim(0, max_clim)
    fig.supylabel("Trial #")
    fig.supxlabel("Time centered on lick/approach to port [s]")
    fig.suptitle(fig_titles[ScrollObj.current_position])
    fig.tight_layout()
    ScrollObj.last_position = len(port_activations)

def plot_daily_rasters(ScrollObj):
    attrs = ["rasters", "tuning_curves", "titles"]
    check_attrs(ScrollObj, attrs)

    axs = ScrollObj.ax
    rasters = ScrollObj.rasters
    tuning_curves = ScrollObj.tuning_curves
    titles = ScrollObj.titles

    for day, (raster, tuning_curve, title) in enumerate(zip(rasters, tuning_curves, titles)):
        axs[day, 0].imshow(raster[ScrollObj.current_position], cmap='gray', aspect='auto')
        axs[day, 0].set_ylabel(f'{title} trials', fontsize=14)
        axs[day, 1].plot(tuning_curve[ScrollObj.current_position])
        axs[day, 1].set_ylabel('$Ca^{2+}$ event '
                               'rate', fontsize=14)
    fig = axs[0,0].figure
    fig.supxlabel('Linearized position (cm)')
    fig.tight_layout()

    ymax = []
    ymin = []
    for ax in axs:
        ylims = ax[1].get_ylim()
        ymin.append(ylims[0])
        ymax.append(ylims[1])

    for ax, raster in zip(axs, rasters):
        ax[1].set_ylim([min(ymin), max(ymax)])
        [ax[1].spines[side].set_visible(False) for side in ['top', 'right']]
        ax[0].set_yticks([1, raster.shape[1]-1])

        for ax_ in ax:
            ax_.set_xticks(ax_.get_xlim())
            ax_.set_xticklabels([0, 220])

    ScrollObj.last_position = rasters[0].shape[0] - 1


def spiral_plot(t, lin_position, markers, ax=None, marker_legend="Licks",
                plot_legend=True):
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
    if markers.ndim == 1:
        markers = markers[np.newaxis, :]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
    ax.plot(lin_position, t)
    for marker in markers:
        ax.plot(lin_position[marker], t[marker], "o", markersize=2)

    if plot_legend:
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

def color_boxes(boxes, colors, alpha=1):
    if type(colors) is str:
        colors = [to_rgba(colors, alpha=alpha) for i in boxes['boxes']]

    for patch, med, color in zip(boxes["boxes"], boxes["medians"], colors):
        patch.set_facecolor(color)
        med.set(color="k")