import numpy as np
import matplotlib.pyplot as plt
import os
from CircleTrack.SessionCollation import MultiAnimal
from CircleTrack.BehaviorFunctions import BehaviorSession
from LinearTrack.MiniscopeFunctions import CalciumSession

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 12})

session_types = ['LinearTrack' + str(i) for i in np.arange(1,6)]

class Drift:
    def __init__(self, mice):
        self.lt_data = MultiAnimal(mice, project_name='LinearTrack',
                                   SessionFunction=CalciumSession,
                                   session_types=session_types)

        self.circle_data = MultiAnimal(mice, project_name='RemoteReversal',
                                       SessionFunction=BehaviorSession)


    def PV_corr(self, mouse, session_type):


if __name__ == '__main__':
    mice = ['Atlas',
            'Miranda',
            'Naiad',
            'Oberon',
            'Puck',
            'Sao',
            'Titania',
            'Umbriel',
            'Virgo',
            'Ymir',
            ]
