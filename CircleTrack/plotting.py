import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import check_attrs

def plot_spiral(ScrollObj):
    attrs = ['t', 'lin_position', 'markers', 'marker_legend']
    check_attrs(ScrollObj, attrs)

    ax = ScrollObj.ax
    lin_position = ScrollObj.lin_position
    t = ScrollObj.t
    this_marker_set = ScrollObj.markers[ScrollObj.current_position]

    ax.plot(lin_position, t)
    ax.plot(lin_position[this_marker_set],
            t[this_marker_set],
            'ro', markersize=2)
    ax.legend(['Trajectory', ScrollObj.marker_legend])

    ax.spines['polar'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ScrollObj.last_position = len(ScrollObj.markers)