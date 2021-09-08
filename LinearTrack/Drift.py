import numpy as np
import matplotlib.pyplot as plt
import os
from CircleTrack.SessionCollation import MultiAnimal
from CircleTrack.BehaviorFunctions import BehaviorSession
from LinearTrack.MiniscopeFunctions import CalciumSession
from CaImaging.Behavior import spatial_bin
from sklearn.impute import SimpleImputer

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 12})

session_types = ['LinearTrack' + str(i) for i in np.arange(1,6)]
directions = ['left', 'right']

class Drift:
    def __init__(self, mice):
        self.lt_data = MultiAnimal(mice, project_name='LinearTrack',
                                   SessionFunction=CalciumSession,
                                   session_types=session_types)

        self.circle_data = MultiAnimal(mice, project_name='RemoteReversal',
                                       SessionFunction=BehaviorSession)

    def PV_corr(self, mouse, session_type,
                nbins=50):
        session = self.lt_data[mouse][session_type]

        # Get x position and running timestamps.
        x = session.behavior.data['df']['x']
        running = session.spatial.data['running']
        session_directions = session.behavior.data['df']['direction']

        # Get imaging data.
        neural_data = session.imaging['S']
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=0)
        neural_data = imp.fit_transform(neural_data.T).T

        pfs = {direction: [] for direction in directions}
        for direction in session_directions:
            mask = np.logical_and(running, session_directions==direction)
            for neuron in neural_data:
                pf = spatial_bin(
                    x[mask],
                    x[mask],
                    nbins=nbins,
                    show_plot=False,
                    weights=neuron[mask],
                    one_dim=True,
                )
                pfs[direction].append(pf)

        return pfs

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
