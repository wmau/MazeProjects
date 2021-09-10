import numpy as np
from matplotlib import pyplot as plt
from CaImaging.util import check_attrs

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
    axs[0].set_ylabel('Laps')
    axs[1].set_xlabel('Linearized position')
    axs[1].set_ylabel('Average transient rate')
    [ax.axis('auto') for ax in axs]

    ScrollObj.last_position = len(ScrollObj.rasters)

def plot_directional_raster(ScrollObj):
    attrs = ["rasters", "tuning_curves"]
    check_attrs(ScrollObj, attrs)

    axs = ScrollObj.ax
    rasters = ScrollObj.rasters
    tuning_curves = ScrollObj.tuning_curves
    fig = ScrollObj.fig

    for i, direction in enumerate(['left', 'right']):
        axs[0, i].imshow(rasters[direction][ScrollObj.current_position], cmap='gray')
        axs[1, i].plot(tuning_curves[direction][ScrollObj.current_position])
        axs[0, i].set_title(f'{direction}ward trials')

    y_max = np.max([axs[1, i].get_ylim()[1] for i in range(2)])
    [axs[1, i].set_ylim(top=y_max) for i in range(2)]

    axs[0, 0].set_ylabel('Trials')
    axs[1, 0].set_ylabel('Average transient rate')
    fig.supxlabel('Linearized position')
    fig.suptitle(ScrollObj.titles[ScrollObj.current_position])
    [axs[0, i].axis('auto') for i in range(2)]

    ScrollObj.last_position = len(ScrollObj.rasters['left'])