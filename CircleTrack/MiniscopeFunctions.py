from CircleTrack.BehaviorFunctions import BehaviorSession
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import sync_data
from util import Session_Metadata

class Session:
    def __init__(self, session_folder):
        self.folder = session_folder
        self.behavior = BehaviorSession(self.folder)

        self.data = dict()
        self.data['behavior'], \
        self.data['imaging'], _ = \
            sync_data(self.behavior.behavior_df,
                      )
        pass

if __name__ == '__main__':
    folder = r'Z:\Will\Drift\Data\Castor_Scope05\09_09_2020_CircleTrackGoals2\16_46_11'
    Session(folder)