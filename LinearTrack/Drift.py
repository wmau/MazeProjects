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
from CaImaging.util import nan_array, sem
from itertools import product
from CaImaging.plotting import jitter_x
import matplotlib.patches as mpatches
import xarray as xr
import pandas as pd
from CaImaging.plotting import errorfill, beautify_ax

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 12})

session_types = {
    'lineartrack': ['LinearTrack' + str(i) for i in np.arange(1,6)],
    'circletrack': ["Goals1", "Goals2", "Goals3", "Goals4", "Reversal"],
}
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
                                   session_types=session_types['lineartrack'])

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

        self.meta["session_labels"] = {
            'circletrack': [session_type.replace("Goals", "Training")
                            for session_type in self.meta["session_types"]['circletrack']],
            'lineartrack': self.meta['session_types']['lineartrack']
    }

    def scatter_box(self, data, ylabel='', ax=None, fig=None,
                    categories=ages, colors=age_colors):
        if ax is None:
            fig, ax = plt.subplots()
        boxes = ax.boxplot([data[i] for i in categories],
                           widths=0.75, showfliers=False, zorder=0, patch_artist=True)

        [ax.scatter(
            jitter_x(np.ones_like(data[category])*(i+1), 0.05),
            data[category],
            color=color,
            edgecolor='k',
            zorder=1,
            s=50,
        )
            for i, (category, color) in enumerate(zip(categories,
                                                      colors))]

        for patch, med, color in zip(
                boxes["boxes"], boxes["medians"], colors
        ):
            patch.set_facecolor(color)
            med.set(color="k")
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        fig.tight_layout()

        patches = [
            mpatches.Patch(facecolor=c, label=label, edgecolor="k")
            for c, label in zip(colors, categories)
        ]
        fig.legend(handles=patches, loc="lower right")

        return fig, ax

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

    def corr_drift_rate_to_behavior(self, corr_matrices,
                                    performance_metric='CRs'):
        drift_rates = self.get_drift_rate(corr_matrices)
        for mouse in self.meta['mice']:
            self.circle_data[mouse]['Reversal'].sdt_trials(
                rolling_window=None,
                plot=False,
            )

        r = dict()
        fig, ax = plt.subplots()
        ylabels = {
            'd_prime': "d'",
            'CRs': 'Correct rejection rate',
            'hits': 'Hit rate',
        }
        for age, color in zip(ages, age_colors):
            performance = [self.circle_data[mouse]['Reversal'].sdt[performance_metric]
                           for mouse in self.meta['grouped_mice'][age]]
            rates = [drift_rates[mouse]
                     for mouse in self.meta['grouped_mice'][age]]

            ax.scatter(rates, performance, color=color)

            r[age] = spearmanr(rates, performance)
        ax.set_xlabel('Drift rate [more negative = more drift]')
        ax.set_ylabel(ylabels[performance_metric])

        performance = [self.circle_data[mouse]['Reversal'].sdt[performance_metric]
                       for mouse in self.meta['mice']]
        rates = [drift_rates[mouse]
                 for mouse in self.meta['mice']]

        r['overall'] = spearmanr(rates, performance)

        return r

    def plot_all_behavior(
            self,
            window=6,
            strides=2,
            ax=None,
            performance_metric="d_prime",
            show_plot=True,
            trial_limit=None,
    ):
        # Preallocate array.
        behavioral_performance_arr = np.zeros(
            (3, len(self.meta["mice"]), len(self.meta["session_types"]['circletrack'])),
            dtype=object,
        )
        categories = ["hits", "CRs", "d_prime"]

        # Fill array with hits/CRs/d' x mouse x session data.
        for j, mouse in enumerate(self.meta["mice"]):
            for k, session_type in enumerate(self.meta["session_types"]['circletrack']):
                session = self.circle_data[mouse][session_type]
                session.sdt_trials(
                    rolling_window=window,
                    trial_interval=strides,
                    plot=False,
                    trial_limit=trial_limit,
                )

                for i, key in enumerate(categories):
                    behavioral_performance_arr[i, j, k] = session.sdt[key]

        # Place into xarray.
        behavioral_performance = xr.DataArray(
            behavioral_performance_arr,
            dims=("metric", "mouse", "session"),
            coords={
                "metric": categories,
                "mouse": self.meta["mice"],
                "session": self.meta["session_types"]['circletrack'],
            },
        )

        # Make awkward array of all mice and sessions. Begin by
        # finding the longest session for each mouse and session.
        longest_sessions = [
            max(
                [
                    len(i)
                    for i in behavioral_performance.sel(
                    metric="d_prime", session=session
                ).values.tolist()
                ]
            )
            for session in self.meta["session_types"]['circletrack']
        ]
        # This will tell us where to draw session borders.
        borders = np.cumsum(longest_sessions)
        borders = np.insert(borders, 0, 0)

        # Get dimensions of new arrays.
        dims = {
            age: (len(self.meta["grouped_mice"][age]), sum(longest_sessions))
            for age in ages
        }
        metrics = {key: nan_array(dims[key]) for key in ages}

        for age in ages:
            for row, mouse in enumerate(self.meta["grouped_mice"][age]):
                for border, session in zip(borders,
                                           self.meta["session_types"]['circletrack']):
                    metric_this_session = behavioral_performance.sel(
                        metric=performance_metric, mouse=mouse, session=session
                    ).values.tolist()
                    length = len(metric_this_session)
                    metrics[age][row, border : border + length] = metric_this_session

        if show_plot:
            ylabels = {
                "d_prime": "d'",
                "CRs": "Correct rejection rate",
                "hits": "Hit rate",
            }
            if ax is None:
                fig, ax = plt.subplots(figsize=(7.4, 5.7))
            else:
                fig = ax.figure

            if window is not None:
                for age, c in zip(ages, age_colors):
                    ax.plot(metrics[age].T, color=c, alpha=0.3)
                    errorfill(
                        range(metrics[age].shape[1]),
                        np.nanmean(metrics[age], axis=0),
                        sem(metrics[age], axis=0),
                        ax=ax,
                        color=c,
                        label=age,
                    )

                for session in borders[1:]:
                    ax.axvline(x=session, color="k")
                ax.set_xticks(borders)
                ax.set_xticklabels(np.insert(longest_sessions, 0, 0))
                ax.set_xlabel("Trial blocks")
                _ = beautify_ax(ax)
            else:
                for age, c in zip(ages, age_colors):
                    ax.plot(metrics[age].T, color=c, alpha=0.3)
                    ax.errorbar(
                        self.meta['session_labels']['circletrack'],
                        np.nanmean(metrics[age], axis=0),
                        sem(metrics[age], axis=0),
                        color=c,
                        label=age,
                        capsize=5,
                        linewidth=3,
                    )
            ax.set_ylabel(ylabels[performance_metric])
            fig.legend()

        if window is None:
            mice_ = np.hstack([np.repeat(self.meta['grouped_mice'][age],
                                         len(self.meta['session_types']['circletrack']))
                               for age in ages])
            ages_ = np.hstack([np.repeat(age, metrics[age].size) for age in ages])
            session_types_ = np.hstack([np.tile(self.meta['session_types']['circletrack'],
                                                len(self.meta['grouped_mice'][age]))
                                        for age in ages])
            metric_ = np.hstack([metrics[age].flatten() for age in ages])

            df = pd.DataFrame(
                {'metric': metric_,
                 'session_types': session_types_,
                 'mice': mice_,
                 'age': ages_,
                 }
            )
        else:
            df = None

        return behavioral_performance, metrics, df


    def get_learning_rate_mouse(self, mouse, performance,
                                performance_metric):
        data = performance.sel(mouse=mouse, session='Reversal',
                               metric=performance_metric).values.tolist()

        rate = spearmanr(range(len(data)), data)[0]

        return rate

    def get_learning_rates(self, performance_metric, window=6, strides=2):
        performance = self.plot_all_behavior(performance_metric=performance_metric,
                                             window=window, strides=strides,
                                             show_plot=False)[0]
        rates = dict()
        for mouse in self.meta['mice']:
            rates[mouse] = self.get_learning_rate_mouse(mouse, performance,
                                                        performance_metric)

        return rates

    def corr_drift_rate_to_learning_rate(self, corr_matrices,
                                         performance_metric,
                                         window=6, strides=2):
        learning_rates = self.get_learning_rates(performance_metric,
                                                 window=window, strides=strides)

        drift_rates_grouped = self.compare_drift_rates(corr_matrices,
                                                       show_plot=False)

        fig, ax = plt.subplots()
        learning_rates_grouped = {
            age: [learning_rates[mouse]
                  for mouse in self.meta['grouped_mice'][age]]
            for age in ages
        }

        ylabel = {
            'hits': 'Reversal hit learning rate',
            'CRs': 'Reversal correct rejection learning rate',
            'd_prime': "Reversal d' learning rate"
        }
        for age, color in zip(ages, age_colors):
            ax.scatter(drift_rates_grouped[age],
                       learning_rates_grouped[age],
                       color=color)
        ax.set_ylabel(ylabel[performance_metric])
        ax.set_xlabel('Drift rate [more negative = more drift]')


        drift_rates = drift_rates_grouped['young'] + drift_rates_grouped['aged']
        learning_rates = learning_rates_grouped['young'] + learning_rates_grouped['aged']

        r, pvalue = spearmanr(drift_rates, learning_rates)

        return r, pvalue

    def compare_drift_rates(self, corr_matrices, show_plot=True):
        drift_rates = self.get_drift_rate(corr_matrices)
        drift_rate_ages = {age: [drift_rates[mouse]
                                 for mouse in self.meta['grouped_mice'][age]] for age in ages}

        if show_plot:
            fig, ax = self.scatter_box(drift_rate_ages,
                                       ylabel='Drift rates '
                                              '[more negative = more drift]')

        return drift_rate_ages

    def PV_corr_all_mice(self, nbins=26, normalize_by_occ=True):
        corr_matrices = dict()
        for mouse in self.meta['mice']:
            print(f'Analyzing {mouse}...')
            corr_matrices[mouse] = self.session_pairwise_PV_corr_efficient(mouse, nbins=nbins,
                                                                 normalize_by_occ=normalize_by_occ)

        return corr_matrices

    def plot_corr_matrix(self, corr_matrices):
        fig, axs = plt.subplots(2,2, figsize=(8,9))

        matrices = []
        for row_ax, age in zip(axs, ages):
            for ax, direction in zip(row_ax, directions):
                matrix = np.nanmean([corr_matrices[mouse][direction]
                                     for mouse in self.meta['grouped_mice'][age]], axis=0)

                matrices.append(matrix)
                ax.imshow(matrix)
                ax.set_title(f'{age}, {direction}ward trials')
                ax.set_xlabel('Day')
                ax.set_ylabel('Day')

        min_clim = np.min(matrices)
        max_clim = np.max(matrices)

        for ax in axs.flatten():
            for im in ax.get_images():
                im.set_clim(min_clim, max_clim)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Mean PV correlation [Spearman rho]')
        #fig.tight_layout()


    def get_diagonals(self, corr_matrices):
        data = {}
        for mouse in self.meta['mice']:
            data[mouse] = {
                'coefs': [],
                'day_lag': [],
            }
            for i in range(len(self.meta['session_types']['lineartrack'])):
                rhos = np.hstack(
                    [np.diag(corr_matrices[mouse][direction], k=i)
                     for direction in directions]
                )

                data[mouse]['coefs'].extend(rhos)
                data[mouse]['day_lag'].extend(np.ones_like(rhos)*i)

            data[mouse]['coefs'] = np.asarray(data[mouse]['coefs'])
            data[mouse]['day_lag'] = np.asarray(data[mouse]['day_lag'])

        return data

    def compare_PV_corrs(self, corr_matrices):
        data = self.get_diagonals(corr_matrices)
        n_sessions = len(self.meta['session_types']['lineartrack'])
        PV_corrs = {}
        for day_lag in range(n_sessions):
            PV_corrs[day_lag] = dict()
            for age in ages:
                PV_corrs[day_lag][age] = []
                for mouse in self.meta['grouped_mice'][age]:
                    coefs = data[mouse]['coefs'][data[mouse]['day_lag'] == day_lag]
                    coefs = coefs[~np.isnan(coefs)]
                    PV_corrs[day_lag][age].extend(coefs)

        fig, axs = plt.subplots(1, n_sessions, sharey=True, figsize=(10.5, 5))
        for day_lag, ax in enumerate(axs):
            self.scatter_box(PV_corrs[day_lag], ax=ax, fig=fig)
            ax.set_title(f'{day_lag} days apart')
        axs[0].set_ylabel('PV correlation [Spearman rho]')
        fig.tight_layout()

        return PV_corrs

    def plot_one_drift_rate(self, mouse, corr_matrices):
        data = self.get_diagonals(corr_matrices)
        fig, ax = plt.subplots()
        ax.scatter(data[mouse]['day_lag'], data[mouse]['coefs'])
        ax.set_xticks(range(len(self.meta['session_types']['lineartrack'])))
        ax.set_xlabel('Day lag')
        ax.set_ylabel('PV correlation [rho]')

    def get_drift_rate(self, corr_matrices):
        data = self.get_diagonals(corr_matrices)
        drift_rates = {mouse: spearmanr(data[mouse]['day_lag'],
                                        data[mouse]['coefs'],
                                        nan_policy='omit')[0]
                       for mouse in self.meta['mice']}

        return drift_rates

    def session_pairwise_PV_corr_efficient(self, mouse,
                                           nbins=26, normalize_by_occ=True,
                                           corr='spearman'):
        pfs = {}
        for session in self.meta['session_types']['lineartrack']:
            try:
                pfs[session] = self.get_directional_pfs(mouse, session, nbins=nbins,
                                                        normalize_by_occ=normalize_by_occ)
            except:
                pass
        corr_fun = spearmanr if corr=='spearman' else pearsonr

        shape = (len(session_types['lineartrack']),
                 len(session_types['lineartrack']))
        corr_matrix = {direction: nan_array(shape) for direction in directions}
        for i, session_pair in enumerate(product(session_types['lineartrack'],
                                                 repeat=2)):
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
        shape = (len(session_types['lineartrack']),
                 len(session_types['lineartrack']))
        corr_matrix = {direction: nan_array(shape) for direction in directions}
        for i, session_pair in enumerate(product(session_types['lineartrack'],
                                                 repeat=2)):
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
        """
        Split the session in half by taking every other trial and computing
        place fields. Note that one trial is in one direction (trial 0 = left,
        trial 1 = right, trial 2 = left), so we taking every 4 trials by
        that definition.
        """
        session = self.lt_data[mouse][session_type]

        # Get relevant behavioral data.
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
        # Figure out which trial to start on for left and right directions.
        start_trial = {
            direction: min(trials[session_directions==direction])
            for direction in directions
        }

        for trial_offset, trial_type in zip([0, 2], ['even', 'odd']):
            for direction in directions:
                trials_to_include = range(start_trial[direction] + trial_offset,
                                          session.behavior.data['ntrials'], 4)

                # Only take running in one direction and in the list of trials.
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
            fig, axs = plt.subplots(2,2, figsize=(8,6))
            for row, direction in zip(axs, directions):
                for ax, trial_type in zip(row, ['even', 'odd']):
                    ax.imshow(pfs[trial_type][direction][orders[direction]], aspect='auto')
                    ax.set_title(f'{direction}ward {trial_type} trials')
            fig.supylabel('Neuron #')
            fig.supxlabel('Linearized position')

        return pfs

    def get_directional_pfs(self, mouse, session_type, nbins=51, neurons=None,
                            normalize_by_occ=True):
        """
        Get place fields for both left and right directions in a single session.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_type: str
            Session name (e.g., 'LinearTrack1').

        nbins: int
            Number of spatial bins.

        neurons: array-like or None
            Neurons to include. If None, include all neurons.

        normalize_by_occ: bool
            Normalize by occupancy.
        """
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
            # Only get running epochs and when moving in one of the two directions.
            mask = np.logical_and(running, session_directions==direction)
            position = x[mask]

            # Get occupancy map.
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

    import pickle as pkl
    with open(r'Z:\Will\RemoteReversal\Data\PV_corr_matrices.pkl', 'rb') as file:
        CT_corr_matrices = pkl.load(file)

    with open(r'Z:\Will\LinearTrack\Data\PV_corr_matrix.pkl', 'rb') as file:
        LT_corr_matrices = pkl.load(file)

    drift_rates_LT = D.get_drift_rate(LT_corr_matrices)
    drift_rates_CT = RR.get_drift_rate(CT_corr_matrices)
    fig, ax = plt.subplots(figsize=(4,5))
    for age, color in zip(['young', 'aged'], ['cornflowerblue', 'r']):
        for mouse in RR.meta['grouped_mice'][age]:
            try:
                ax.plot([0, 1], [drift_rates_LT[mouse], drift_rates_CT[mouse]], c=color)
            except:
                ax.scatter(1, drift_rates_CT[mouse], c=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Linear track', 'Circle track'])
    ax.set_ylabel('Drift rate')
    fig.tight_layout()