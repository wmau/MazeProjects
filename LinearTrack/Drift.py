import numpy as np
import matplotlib.pyplot as plt
import os
from CircleTrack.SessionCollation import MultiAnimal
from CircleTrack.BehaviorFunctions import BehaviorSession
from LinearTrack.MiniscopeFunctions import CalciumSession
from CaImaging.Behavior import spatial_bin
from sklearn.impute import SimpleImputer
from CaImaging.CellReg import rearrange_neurons, trim_map
from scipy.stats import spearmanr, pearsonr
from CaImaging.util import nan_array
from itertools import product

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 12})

session_types = ['LinearTrack' + str(i) for i in np.arange(1,6)]
directions = ['left', 'right']
ages = ["young", "aged"]
age_colors = ["cornflowerblue", "r"]

aged_mice = [
    "Gemini",
    "Oberon",
    "Puck",
    "Umbriel",
    "Virgo",
    "Ymir",
    "Atlas",
    "PSAM_1",
    "PSAM_2",
    "PSAM_3",
]

class Drift:
    def __init__(self, mice):
        self.lt_data = MultiAnimal(mice, project_name='LinearTrack',
                                   SessionFunction=CalciumSession,
                                   session_types=session_types)

        self.circle_data = MultiAnimal(mice, project_name='RemoteReversal',
                                       SessionFunction=BehaviorSession)

        self.meta = {
            "session_types": session_types,
            "mice": mice,
        }

        self.meta["grouped_mice"] = {
            "aged": [mouse for mouse in self.meta["mice"] if mouse in aged_mice],
            "young": [mouse for mouse in self.meta["mice"] if mouse not in aged_mice],
        }

        self.meta["aged"] = {
            mouse: True if mouse in aged_mice else False for mouse in self.meta["mice"]
        }

    def rearrange_neurons(self, mouse, session_types, data_type, detected="everyday"):
        """
        Rearrange neural activity matrix to have the same neuron in
        each row.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_types: array-like of strs
            Must correspond to at least two of the sessions in the
            session_types class attribute (e.g. 'CircleTrackGoals2'.

        data_type: str
            Neural activity data type that you want to align. (e.g.
            'S' or 'C' or 'patterns').
        """
        sessions = self.lt_data[mouse]
        trimmed_map = self.get_cellreg_mappings(
            mouse, session_types, detected=detected, neurons_from_session1=None
        )[0]

        # Get calcium activity from each session for this mouse.
        if data_type == "patterns":
            activity_list = [
                sessions[session].assemblies[data_type].T for session in session_types
            ]
        else:
            activity_list = [
                sessions[session].imaging[data_type] for session in session_types
            ]

        # Rearrange the neurons.
        rearranged = rearrange_neurons(trimmed_map, activity_list)

        return rearranged

    def get_cellreg_mappings(
            self, mouse, session_types, detected="everyday", neurons_from_session1=None
    ):
        # For readability.
        cellreg_map = self.lt_data[mouse]["CellReg"].map
        cellreg_sessions = self.lt_data[mouse]["CellReg"].sessions

        # Get the relevant list of session names that the CellReg map
        # dataframe recognizes. This is probably not efficient but the
        # order of list elements works this way.
        session_list = []
        for session_type in session_types:
            for session in cellreg_sessions:
                if session_type in session:
                    session_list.append(session)

        trimmed_map = trim_map(cellreg_map, session_list, detected=detected)

        if neurons_from_session1 is None:
            global_idx = trimmed_map.index
        else:
            in_list = trimmed_map.iloc[:, 0].isin(neurons_from_session1)
            global_idx = trimmed_map[in_list].index

        return trimmed_map, global_idx, session_list

    def PV_corr_all_mice(self, nbins=26, normalize_by_occ=True):
        corr_matrices = dict()
        for mouse in self.meta['mice']:
            print(f'Analyzing {mouse}...')
            corr_matrices[mouse] = self.session_pairwise_PV_corr(mouse, nbins=nbins,
                                                                 normalize_by_occ=normalize_by_occ)

        return corr_matrices

    def PV_corr_all_mice_efficient(self, nbins=26, normalize_by_occ=True):
        corr_matrices = dict()
        for mouse in self.meta['mice']:
            print(f'Analyzing {mouse}...')
            corr_matrices[mouse] = self.session_pairwise_PV_corr_efficient(mouse, nbins=nbins,
                                                                 normalize_by_occ=normalize_by_occ)

        return corr_matrices


    def session_pairwise_PV_corr_efficient(self, mouse,
                                           nbins=26, normalize_by_occ=True,
                                           corr='spearman'):
        pfs = {}
        for session in self.meta['session_types']:
            try:
                pfs[session] = self.get_directional_pfs(mouse, session, nbins=nbins,
                                                        normalize_by_occ=normalize_by_occ)
            except:
                pass
        corr_fun = spearmanr if corr=='spearman' else pearsonr

        shape = (len(session_types), len(session_types))
        corr_matrix = {direction: nan_array(shape) for direction in directions}
        for i, session_pair in enumerate(product(session_types, repeat=2)):
            same_session = session_pair[0] == session_pair[1]
            row, col = np.unravel_index(i, shape)

            if same_session:
                try:
                    split_pfs = self.get_split_trial_pfs(mouse, session_pair[0], nbins=nbins,
                                                         normalize_by_occ=normalize_by_occ)

                    for direction in directions:
                        even, odd = [split_pfs[trial_type][direction].T
                                     for trial_type in ['even', 'odd']]

                        rhos = []
                        for x, y in zip(even, odd):
                            rhos.append(corr_fun(x,y)[0])

                        corr_matrix[direction][row, col] = np.nanmean(rhos)
                except:
                    pass
            else:
                try:
                    trimmed_map = np.asarray(self.get_cellreg_mappings(mouse, session_pair, detected='everyday')[0])

                    for direction in directions:
                        s1, s2 = [pfs[session][direction][neurons].T
                                  for neurons, session in zip(trimmed_map.T, session_pair)]

                        rhos = []
                        for x,y in zip(s1, s2):
                            rhos.append(corr_fun(x,y)[0])

                        corr_matrix[direction][row, col] = np.nanmean(rhos)

                except:
                    pass

        return corr_matrix


    def session_pairwise_PV_corr(self, mouse,
                                 nbins=51,
                                 normalize_by_occ=True,
                                 corr='spearman'):
        shape = (len(session_types), len(session_types))
        corr_matrix = {direction: nan_array(shape) for direction in directions}
        for i, session_pair in enumerate(product(session_types, repeat=2)):
            try:
                rhos = self.PV_corr(mouse, session_pair, nbins=nbins, normalize_by_occ=normalize_by_occ, corr=corr)

                row, col = np.unravel_index(i, shape)

                for direction in directions:
                    corr_matrix[direction][row, col] = np.nanmean(rhos[direction])
            except:
                pass

        return corr_matrix

    def PV_corr(self, mouse,
                sessions,
                nbins=51,
                normalize_by_occ=True,
                corr='spearman'):
        same_session = sessions[0] == sessions[1]
        if same_session:
            pfs = self.get_split_trial_pfs(mouse, sessions[0],
                                           nbins=nbins,
                                           normalize_by_occ=normalize_by_occ)
        else:
            pfs = self.align_pfs(mouse, sessions, nbins=nbins, normalize_by_occ=normalize_by_occ)[0]

        rhos = {}
        for direction in directions:
            rhos[direction] = []

            if same_session:
                s1, s2 = [pfs[trials][direction].T for trials in ['even', 'odd']]
            else:
                s1, s2 = [pfs[session][direction].T for session in sessions]

            if corr == 'spearman':
                for x, y in zip(s1, s2):
                    rhos[direction].append(spearmanr(x, y)[0])

            elif corr == 'pearson':
                for x,y in zip(s1, s2):
                    rhos[direction].append(pearsonr(x,y)[0])

        return rhos

    def align_pfs(self, mouse, sessions, nbins=51, normalize_by_occ=True):
        trimmed_map = np.asarray(self.get_cellreg_mappings(mouse, sessions, detected='everyday')[0])

        pfs = {
            session: self.get_directional_pfs(mouse, session, nbins=nbins,
                                              normalize_by_occ=normalize_by_occ,
                                              neurons=neurons)
            for session, neurons in zip(sessions, trimmed_map.T)
        }

        peaks = {direction: np.argmax(pfs[sessions[0]][direction], axis=1) for direction in directions}
        orders = {direction: np.argsort(peaks[direction]) for direction in directions}

        return pfs, orders

    def get_split_trial_pfs(self, mouse, session_type, nbins=51, neurons=None,
                            normalize_by_occ=True, show_plot=False):
        session = self.lt_data[mouse][session_type]

        x = session.behavior.data['df']['x']
        running = session.spatial.data['running']
        session_directions = session.behavior.data['df']['direction']
        trials = session.behavior.data['df']['trials']

        # Get imaging data.
        neural_data = session.imaging['S']
        imp = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=0)
        neural_data = imp.fit_transform(neural_data.T).T

        # Only analyze selected neurons, unless unspecified.
        if neurons is None:
            neurons = [n for n in range(neural_data.shape[0])]
        neural_data = neural_data[neurons]

        pfs = {
            trial: {direction: [] for direction in directions}
            for trial in ['even', 'odd']
        }
        start_trial = {
            direction: min(trials[session_directions==direction])
            for direction in directions
        }

        for trial_offset, trial_type in zip([0, 2], ['even', 'odd']):
            for direction in directions:
                trials_to_include = range(start_trial[direction] + trial_offset,
                                          session.behavior.data['ntrials'], 4)
                mask = np.logical_and(running, session_directions==direction)
                mask = np.logical_and(mask, np.in1d(trials, trials_to_include))
                position = x[mask]

                if normalize_by_occ:
                    occupancy_map = spatial_bin(
                        position,
                        position,
                        nbins=nbins,
                        one_dim=True,
                    )[0]

                for neuron in neural_data:
                    pf = spatial_bin(
                        position,
                        position,
                        nbins=nbins,
                        show_plot=False,
                        weights=neuron[mask],
                        one_dim=True
                    )[0]

                    if normalize_by_occ:
                        pf = pf / occupancy_map

                    pfs[trial_type][direction].append(pf)

                pfs[trial_type][direction] = np.vstack(pfs[trial_type][direction])

        if show_plot:
            orders = {direction: np.argsort(np.argmax(pfs['even'][direction], axis=1))
                      for direction in directions}
            fig, axs = plt.subplots(2,2)
            for row, direction in zip(axs, directions):
                for ax, trial_type in zip(row, ['even', 'odd']):
                    ax.imshow(pfs[trial_type][direction][orders[direction]], aspect='auto')

        return pfs

    def get_directional_pfs(self, mouse, session_type, nbins=51, neurons=None,
                            normalize_by_occ=True):
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

        # Only analyze selected neurons, unless unspecified.
        if neurons is None:
            neurons = [n for n in range(neural_data.shape[0])]
        neural_data = neural_data[neurons]

        pfs = {direction: [] for direction in directions}
        for direction in directions:
            mask = np.logical_and(running, session_directions==direction)
            position = x[mask]

            if normalize_by_occ:
                occupancy_map = spatial_bin(
                    position,
                    position,
                    nbins=nbins,
                    one_dim=True,
                )[0]

            for neuron in neural_data:
                pf = spatial_bin(
                    position,
                    position,
                    nbins=nbins,
                    show_plot=False,
                    weights=neuron[mask],
                    one_dim=True
                )[0]

                if normalize_by_occ:
                    pf = pf / occupancy_map

                pfs[direction].append(pf)

            pfs[direction] = np.vstack(pfs[direction])

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
    D = Drift(mice)
    D.session_pairwise_PV_corr_efficient('Ymir')