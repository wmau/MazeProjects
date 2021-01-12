import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import check_attrs


def plot_spiral(ScrollObj):
    attrs = ["t", "lin_position", "markers", "marker_legend"]
    check_attrs(ScrollObj, attrs)

    ax = ScrollObj.ax
    lin_position = ScrollObj.lin_position
    t = ScrollObj.t
    this_marker_set = ScrollObj.markers[ScrollObj.current_position]

    ax.plot(lin_position, t)
    ax.plot(lin_position[this_marker_set], t[this_marker_set], "ro", markersize=2)
    ax.legend(["Trajectory", ScrollObj.marker_legend])

    ax.spines["polar"].set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ScrollObj.last_position = len(ScrollObj.markers)


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
    axs[0].set_aspect(5)
    axs[1].plot(tuning_curve[ScrollObj.current_position])

    ScrollObj.last_position = len(ScrollObj.rasters)


def plot_daily_rasters(ScrollObj):
    attrs = ["rasters"]
    check_attrs(ScrollObj, attrs)

    axs = ScrollObj.ax
    rasters = ScrollObj.rasters
    # tuning_curves = ScrollObj.tuning_curves

    for day, raster in enumerate(rasters):
        axs[day, 0].imshow(raster[ScrollObj.current_position])

    ScrollObj.last_position = rasters[0].shape[0]
