import tkinter as tk
tkroot = tk.Tk()
tkroot.withdraw()
from tkinter import filedialog
import pandas as pd

metadata_csv = r'Z:\Will\Drift\Pilots\Metadata.csv'

class Preprocess:
    def __init__(self, metadata_csv=metadata_csv, folder=None):
        """

        :param folder:
        """
        if folder is None:
            self.folder = filedialog.askdirectory()
        else:
            self.folder = folder

        metadata_df = pd.read_csv(metadata_csv)

        entry_match = metadata_df.loc[metadata_df['Path'] == folder]


        pass



if __name__ == '__main__':
    folder = r'Z:\Will\Drift\Pilots\M1\07_12_2020_TMazeFreeChoice2\H15_M23_S4'
    Preprocess(folder=folder)