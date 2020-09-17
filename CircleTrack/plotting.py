import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import check_attrs

def plot_spiral(ScrollObj):
    attrs = ['t', 'position', 'markers']
    checkattrs(ScrollObj, attrs)

