import tkinter as tk
tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog

class Preprocess:
    def __init__(self, folder=None, behav_cam=2, miniscope_cam=6):
        """

        :param folder:
        :param behav_cam:
        :param miniscope_cam:
        """

        if folder is None:
            self.folder = filedialog.askdirectory()
        else:
            self.folder = folder