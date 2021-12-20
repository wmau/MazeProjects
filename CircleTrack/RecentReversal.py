import matplotlib.pyplot as plt
from CaImaging.util import (
    sem,
    nan_array,
    bin_transients,
    make_bins,
    ScrollPlot,
    contiguous_regions,
    cluster,
    stack_padding,
    distinct_colors,
)
from CaImaging.plotting import errorfill, beautify_ax, jitter_x
from scipy.stats import (
    spearmanr,
    zscore,
    circmean,
    kendalltau,
    wilcoxon,
    mannwhitneyu,
    pearsonr,
)
from scipy.spatial import distance
from CircleTrack.SessionCollation import MultiAnimal
from CircleTrack.MiniscopeFunctions import CalciumSession
from CaImaging.CellReg import rearrange_neurons, trim_map, scrollplot_footprints
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    permutation_test_score,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import numpy as np
import os
from CircleTrack.plotting import (
    plot_daily_rasters,
    spiral_plot,
    highlight_column,
    plot_port_activations,
    color_boxes,
)
from CaImaging.Assemblies import find_assemblies, preprocess_multiple_sessions, lapsed_activation
from CircleTrack.Assemblies import (
    plot_assembly,
    spatial_bin_ensemble_activations,
    find_members,
    find_memberships,
    plot_pattern,
)
import xarray as xr
import pymannkendall as mk
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product, cycle, islice
from CaImaging.PlaceFields import spatial_bin, PlaceFields
from tqdm import tqdm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
import pickle as pkl
from CircleTrack.utils import (
    get_circular_error,
    format_spatial_location_for_decoder,
    get_equivalent_local_path,
    find_reward_spatial_bins,
)
import pandas as pd
import pingouin as pg

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 12})

session_types = {
    "Drift": [
        "CircleTrackShaping1",
        "CircleTrackShaping2",
        "CircleTrackGoals1",
        "CircleTrackGoals2",
        "CircleTrackReversal1",
        "CircleTrackReversal2",
        "CircleTrackRecall",
    ],
    "RemoteReversal": ["Goals1", "Goals2", "Goals3", "Goals4", "Reversal"],
    "PSAMReversal": ["Goals1", "Goals2", "Goals3", "Goals4", "Reversal"],
}

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

PSEM_mice = ["PSAM_2", "PSAM_3", "PSAM_5", "PSAM_6", "PSAM_7", "PSAM_8", "PSAM_10"]

ages = ["young", "aged"]
PSAM_groups = ["vehicle", "PSEM"]
age_colors = ["cornflowerblue", "r"]
PSAM_colors = ["silver", "coral"]


class RecentReversal:
    def __init__(self, mice, project_name="RemoteReversal", behavior_only=False):
        # Collect data from all mice and sessions.
        self.data = MultiAnimal(
            mice,
            project_name=project_name,
            SessionFunction=CalciumSession,
        )

        # Define session types here. Watch out for typos.
        # Order matters. Plots will be in the order presented here.
        self.meta = {
            "session_types": session_types[project_name],
            "mice": mice,
        }

        self.meta["session_labels"] = [
            session_type.replace("Goals", "Training")
            for session_type in self.meta["session_types"]
        ]

        self.meta["grouped_mice"] = {
            "aged": [mouse for mouse in self.meta["mice"] if mouse in aged_mice],
            "young": [mouse for mouse in self.meta["mice"] if mouse not in aged_mice],
        }

        self.meta["aged"] = {
            mouse: True if mouse in aged_mice else False for mouse in self.meta["mice"]
        }

        # Get spatial fields of the assemblies.
        if not behavior_only:
            for mouse in self.meta["mice"]:
                for session_type in self.meta["session_types"]:
                    session = self.data[mouse][session_type]
                    folder = session.meta["folder"]

                    if session.meta["local"]:
                        folder = get_equivalent_local_path(folder)
                    fpath = os.path.join(folder, "AssemblyFields.pkl")

                    try:
                        with open(fpath, "rb") as file:
                            session.assemblies["fields"] = pkl.load(file)
                    except:
                        behavior = session.behavior.data["df"]
                        session.assemblies["fields"] = PlaceFields(
                            np.asarray(behavior["t"]),
                            np.asarray(behavior["x"]),
                            np.asarray(behavior["y"]),
                            session.assemblies["activations"],
                            bin_size=session.spatial.meta["bin_size"],
                            circular=True,
                            shuffle_test=True,
                            fps=session.spatial.meta["fps"],
                            velocity_threshold=0,
                        )

                        with open(fpath, "wb") as file:
                            pkl.dump(session.assemblies["fields"], file)

    ############################ HELPER FUNCIONS ############################
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
        sessions = self.data[mouse]
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

    def plot_registered_cells(self, mouse, session_types, neurons_from_session1=None):
        session_list = self.get_cellreg_mappings(
            mouse, session_types, neurons_from_session1=neurons_from_session1
        )[-1]

        scrollplot_footprints(
            self.data[mouse]["CellReg"].path, session_list, neurons_from_session1
        )

    def get_cellreg_mappings(
        self, mouse, session_types, detected="everyday", neurons_from_session1=None
    ):
        # For readability.
        cellreg_map = self.data[mouse]["CellReg"].map
        cellreg_sessions = self.data[mouse]["CellReg"].sessions

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

    def set_age_legend(self, fig):
        patches = [
            mpatches.Patch(facecolor=c, label=label, edgecolor="k")
            for c, label in zip(age_colors, ages)
        ]
        fig.legend(handles=patches, loc="lower right")

    def plot_neuron_count(self, sessions_to_plot=None):
        """
        Plot the number of neurons in each session.

        :param sessions_to_plot:
        :return:
        """
        # Get sessions and labels.
        if sessions_to_plot is None:
            sessions_to_plot = self.meta['session_types']
        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in sessions_to_plot
        ]

        # Get number of neurons.
        n_neurons = pd.DataFrame(index=self.meta['mice'])
        for session in sessions_to_plot:
            n_neurons[session] = [self.data[mouse][session].imaging['n_neurons']
                                  for mouse in self.meta['mice']]

        n_neurons['aged'] = [self.meta['aged'][mouse] for mouse in n_neurons.index]

        fig, axs = plt.subplots(1, len(sessions_to_plot),
                                sharey=True)
        fig.subplots_adjust(wspace=0)
        for ax, session, session_label in zip(axs, sessions_to_plot, session_labels):
            data = dict()
            for age in ages:
                aged = age == 'aged'
                data[age] = n_neurons[session].loc[n_neurons['aged'] == aged]

            self.scatter_box(data, ax=ax)
            ax.set_title(session_label)
        axs[0].set_ylabel('Number of neurons')
        self.set_age_legend(fig)

        n_neurons = n_neurons.sort_values(by='aged')
        return n_neurons

    ############################ BEHAVIOR FUNCTIONS ############################

    def plot_licks(self, mouse, session_type, binarize=True):
        """
        Plot the lick matrix (laps x port) for a given mouse and session. Highlight the correct reward location in green.
        If the session is Reversal, also highlight Goals4 reward location in orange.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_type: str
            Session name (e.g. 'Goals4').

        """
        ax = self.data[mouse][session_type].behavior.get_licks(
            plot=True, binarize=binarize
        )[1]
        if session_type == "Reversal":
            [
                highlight_column(rewarded, ax, linewidth=5, color="orange", alpha=0.6)
                for rewarded in np.where(
                    self.data[mouse]["Goals4"].behavior.data["rewarded_ports"]
                )[0]
            ]
        ax.set_title(f"{mouse}, {session_type} session")
        plt.tight_layout()

        return ax

    def plot_behavior(self, mouse, window=8, strides=2, show_plot=True, ax=None):
        """
        Plot behavioral performance (hits, correct rejections, d') for
        one mouse.

        :parameters
        ---
        mouse: str
            Name of the mouse.

        window: int
            Number of trials to calculate hit rate, correct rejection rate,
            d' over.

        strides: int
            Number of overlapping trials between windows.

        show_plot: bool
            Whether or not to plot the individual mouse data.

        ax: Axes object
        """
        # Preallocate dict.
        categories = ["hits", "CRs", "d_prime"]
        sdt = {key: [] for key in categories}
        sdt["session_borders"] = [0]
        sdt["n_trial_blocks"] = [0]

        # For each session, get performance and append it to an array
        # to put in one plot.
        for session_type in self.meta["session_types"]:
            session = self.data[mouse][session_type].behavior
            session.sdt_trials(
                rolling_window=window, trial_interval=strides, plot=False
            )
            for key in categories:
                sdt[key].extend(session.sdt[key])

            # Length of this session (depends on window and strides).
            sdt["n_trial_blocks"].append(len(session.sdt["hits"]))

        # Session border, for plotting a line to separate
        # sessions.
        sdt["session_borders"] = np.cumsum(sdt["n_trial_blocks"])

        # Plot.
        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax2 = ax.twinx()
            ax.plot(sdt["d_prime"], color="k", label="d'")
            for key, c in zip(categories[:-1], ["g", "b"]):
                ax2.plot(sdt[key], color=c, alpha=0.3, label=key)

            ax.set_ylabel("d'")
            for session in sdt["session_borders"][1:]:
                ax.axvline(x=session, color="k")
            ax.set_xticks(sdt["session_borders"])
            ax.set_xticklabels(sdt["n_trial_blocks"])

            ax2.set_ylabel("Hit/Correct rejection rate", rotation=270, labelpad=10)
            ax.set_xlabel("Trial blocks")
            fig.legend()

        return sdt, ax

    def plot_all_behavior(
        self,
        window=6,
        strides=2,
        ax=None,
        performance_metric="d_prime",
        show_plot=True,
        trial_limit=None,
    ):
        """
        Plot behavior metrics for all mice, separated by aged versus young.

        """
        # Preallocate array.
        behavioral_performance_arr = np.zeros(
            (3, len(self.meta["mice"]), len(self.meta["session_types"])),
            dtype=object,
        )
        categories = ["hits", "CRs", "d_prime"]

        # Fill array with hits/CRs/d' x mouse x session data.
        for j, mouse in enumerate(self.meta["mice"]):
            for k, session_type in enumerate(self.meta["session_types"]):
                session = self.data[mouse][session_type].behavior
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
                "session": self.meta["session_types"],
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
            for session in self.meta["session_types"]
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
                for border, session in zip(borders, self.meta["session_types"]):
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
                        self.meta["session_labels"],
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
            mice_ = np.hstack(
                [
                    np.repeat(
                        self.meta["grouped_mice"][age], len(self.meta["session_types"])
                    )
                    for age in ages
                ]
            )
            ages_ = np.hstack([np.repeat(age, metrics[age].size) for age in ages])
            session_types_ = np.hstack(
                [
                    np.tile(
                        self.meta["session_types"], len(self.meta["grouped_mice"][age])
                    )
                    for age in ages
                ]
            )
            metric_ = np.hstack([metrics[age].flatten() for age in ages])

            df = pd.DataFrame(
                {
                    "metric": metric_,
                    "session_types": session_types_,
                    "mice": mice_,
                    "age": ages_,
                }
            )
        else:
            df = None

        return behavioral_performance, metrics, df

    def behavior_anova(self, performance_metric="CRs"):
        """
        Plot behavioral metrics for all mice, separated into aged versus young.
        window=None, meaning data is not smoothed across trials and is aggregated across the whole session.
        Also returns ANOVA results.

        :parameter
        ---
        performance_metric: "d_prime", "CRs", or "hits"

        :returns
        ---
        anova_df: DataFrame
            Results from a mixed ANOVA.

        pairwise_df: DataFrame
            Results from pairwise t-tests.
        """
        df = self.plot_all_behavior(
            performance_metric=performance_metric, window=None, strides=None
        )[2]

        anova_df = pg.mixed_anova(
            df, dv="metric", within="session_types", between="age", subject="mice"
        )

        pairwise_df = df.pairwise_ttests(
            dv="metric",
            between="age",
            within="session_types",
            subject="mice",
            padjust="none",
        )
        return anova_df, pairwise_df, df

    def plot_performance_session_type(
        self,
        session_type,
        ax=None,
        window=None,
        strides=None,
        performance_metric="d_prime",
        show_plot=True,
        downsample_trials=False,
    ):
        """
        Plot the performance of all mice, separated by age, on that session.

        :parameters
        ---
        session_type: str

        :returns
        ---
        peak_performance: dict
            Behavioral performance split into young versus aged.

        """
        if downsample_trials:
            trial_limit = min(
                [
                    self.data[mouse][session_type].behavior.data["ntrials"]
                    for mouse in self.meta["mice"]
                ]
            )
        else:
            trial_limit = None

        behavioral_performance = self.plot_all_behavior(
            show_plot=False,
            window=window,
            strides=strides,
            performance_metric=performance_metric,
            trial_limit=trial_limit,
        )[0]

        peak_performance = dict()
        for age in ages:
            peak_performance[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                peak_performance[age].append(
                    np.nanmax(
                        behavioral_performance.sel(
                            mouse=mouse, metric=performance_metric, session=session_type
                        ).values.tolist()
                    )
                )

        if show_plot:
            label_axes = True
            ylabels = {
                "CRs": "Correct rejection rate",
                "hits": "Hit rate",
                "d_prime": "d'",
            }
            if ax is None:
                fig, ax = plt.subplots(figsize=(3, 4.75))
            else:
                label_axes = False

            self.scatter_box(peak_performance, ax=ax)

            if label_axes:
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["Young", "Aged"])
                ax.set_ylabel(ylabels[performance_metric])
                # ax = beautify_ax(ax)
                plt.tight_layout()

        return peak_performance

    def peak_perf_to_df(self, peak_performance, label='CRs'):
        """
        Convert peak_performance from plot_best_performance() into
        a df.

        :param peak_performance:
        :param label:
        :return:
        """
        df = pd.DataFrame(np.hstack(
            (peak_performance[age]
             for age in ages)),
            index=np.hstack([self.meta['grouped_mice'][age]
                             for age in ages]),
        columns=[label])

        df['aged'] = [self.meta['aged'][mouse] for mouse in df.index]

        return df


    def plot_peak_performance_all_sessions(
        self,
        window=None,
        strides=None,
        performance_metric="CRs",
        downsample_trials=False,
        sessions=None,
    ):
        """
        Plot the peak performance for each session, either when smoothing across trial windows, or when
        window=None, simply the behavioral performance across the whole session.

        :parameters
        ---
        window: int or None
            Trial window size to compute behavioral metrics across.

        strides: int or None
            Overlap between trial windows.

        sessions: list of str
            Sessions to plot.

        """
        if sessions is None:
            sessions = self.meta["session_types"]

        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in sessions
        ]

        fig, axs = plt.subplots(1, len(sessions), sharey=True)
        fig.subplots_adjust(wspace=0)
        ylabels = {
            "d_prime": "d'",
            "CRs": "Correct rejection rate",
            "hits": "Hit rate",
        }
        performance = dict()
        for ax, session, title in zip(axs, sessions, session_labels):
            performance[session] = self.plot_performance_session_type(
                session_type=session,
                ax=ax,
                window=window,
                strides=strides,
                performance_metric=performance_metric,
                downsample_trials=downsample_trials,
            )
            ax.set_xticks([])
            ax.set_title(title)
        axs[0].set_ylabel(ylabels[performance_metric])

        patches = [
            mpatches.Patch(facecolor=c, label=label, edgecolor="k")
            for c, label in zip(age_colors, ages)
        ]
        fig.legend(handles=patches, loc="lower right")

        return performance

    def performance_to_df(self, performance):
        """
        Convert output of plot_peak_performance_all_sessions() to df.

        :param performance:
        :return:
        """
        df = pd.DataFrame()
        for session in performance.keys():
            df[session] = pd.DataFrame(np.hstack(
                (performance[session][age]
                 for age in ages)),
                index=np.hstack([self.meta['grouped_mice'][age]
                                 for age in ages]))

        df['aged'] = [self.meta['aged'][mouse] for mouse in df.index]

        return df

    def scatter_box(self, data, ylabel="", ax=None):
        """
        Make boxplot split into aged versus young. Scatter plot individual data points on top.

        :parameters
        ---
        data: dict
            Data in array-likes, split into "young" and "old".

        """
        if ax is None:
            fig, ax = plt.subplots()
        boxes = ax.boxplot(
            [data[age] for age in ages],
            widths=0.75,
            showfliers=False,
            zorder=0,
            patch_artist=True,
        )

        [
            ax.scatter(
                jitter_x(np.ones_like(data[age]) * (i + 1), 0.05),
                data[age],
                color=color,
                edgecolor="k",
                zorder=1,
                s=50,
            )
            for i, (age, color) in enumerate(zip(ages, age_colors))
        ]

        for patch, med, color in zip(boxes["boxes"], boxes["medians"], age_colors):
            patch.set_facecolor(color)
            med.set(color="k")
        ax.set_xticks([])
        ax.set_ylabel(ylabel)

    def plot_perseverative_licking(self, show_plot=True, binarize=True):
        """
        Plot perseverative versus unforgiveable errors (errors on never-rewarded sites)

        """
        goals4 = "Goals4"
        reversal = "Reversal"

        perseverative_errors = dict()
        unforgiveable_errors = dict()
        for age in ages:
            perseverative_errors[age] = []
            unforgiveable_errors[age] = []

            for mouse in self.meta["grouped_mice"][age]:
                behavior_data = self.data[mouse][reversal].behavior.data

                if binarize:
                    licks = behavior_data["all_licks"] > 0
                else:
                    licks = behavior_data["all_licks"]

                previous_reward_ports = self.data[mouse][goals4].behavior.data[
                    "rewarded_ports"
                ]
                current_rewarded_ports = behavior_data["rewarded_ports"]
                other_ports = ~(previous_reward_ports + current_rewarded_ports)

                perseverative_errors[age].append(
                    np.mean(licks[:, previous_reward_ports])
                )
                unforgiveable_errors[age].append(np.mean(licks[:, other_ports]))

        if show_plot:
            fig, axs = plt.subplots(1, 2, sharey=True)
            fig.subplots_adjust(wspace=0)
            ylabel = {True: "Proportion of trials", False: "Mean licks per trial"}
            for ax, rate, title in zip(
                axs,
                [perseverative_errors, unforgiveable_errors],
                ["Perseverative errors", "Unforgiveable errors"],
            ):
                self.scatter_box(rate, ax=ax)
                ax.set_title(title)
                ax.set_xticks([])

            axs[0].set_ylabel(ylabel[binarize])
            self.set_age_legend(fig)

        return perseverative_errors, unforgiveable_errors

    def plot_perseverative_licking_over_session(
        self,
        session_type="Reversal",
        window_size=8,
        trial_interval=3,
        show_plot=True,
        binarize_licks=True,
    ):
        """
        Plot perseverative and unforgiveable errors averaged across trial windows


        """
        perseverative_errors = dict()
        unforgiveable_errors = dict()
        for age in ages:
            perseverative_errors[age] = [[] for mouse in self.meta["grouped_mice"][age]]
            unforgiveable_errors[age] = [[] for mouse in self.meta["grouped_mice"][age]]

            for i, mouse in enumerate(self.meta["grouped_mice"][age]):
                behavior = self.data[mouse][session_type].behavior
                behavior_data = behavior.data

                licks = behavior.rolling_window_licks(window_size, trial_interval)
                if binarize_licks:
                    licks = licks > 0

                # Find previously rewarded, currently rewarded, and never
                # rewarded ports.
                previous_reward_ports = self.data[mouse]["Goals4"].behavior.data[
                    "rewarded_ports"
                ]
                current_rewarded_ports = behavior_data["rewarded_ports"]
                other_ports = ~(previous_reward_ports + current_rewarded_ports)
                n_previous = np.sum(previous_reward_ports)
                n_other = np.sum(other_ports)

                for licks_this_window in licks:
                    # Get perserverative errors.
                    perseverative_errors[age][i].append(
                        np.sum(licks_this_window[:, previous_reward_ports])
                        / (n_previous * licks_this_window.shape[0])
                    )

                    # Get unforgiveable errors.
                    unforgiveable_errors[age][i].append(
                        np.sum(licks_this_window[:, other_ports])
                        / (n_other * licks_this_window.shape[0])
                    )

        perseverative_errors = {
            age: stack_padding(perseverative_errors[age]) for age in ages
        }
        unforgiveable_errors = {
            age: stack_padding(unforgiveable_errors[age]) for age in ages
        }

        if show_plot:
            ylabel = "Error rate" if binarize_licks else "Average number of licks"
            if session_type == "Reversal":
                fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
                fig.subplots_adjust(wspace=0)

                for ax, rate, title in zip(
                    axs,
                    [perseverative_errors, unforgiveable_errors],
                    ["Perseverative errors", "Unforgiveable errors"],
                ):
                    se = {age: sem(rate[age], axis=0) for age in ages}
                    m = {age: np.nanmean(rate[age], axis=0) for age in ages}
                    for c, age in zip(age_colors, ages):
                        ax.plot(rate[age].T, color=c, alpha=0.1)
                        errorfill(
                            range(m[age].shape[0]), m[age], se[age], color=c, ax=ax
                        )
                        ax.set_title(title)
                    fig.supxlabel("Trial blocks")
                    fig.supylabel(ylabel)
            else:
                fig, ax = plt.subplots()
                se = {age: sem(unforgiveable_errors[age], axis=0) for age in ages}
                m = {age: np.nanmean(unforgiveable_errors[age], axis=0) for age in ages}
                for c, age in zip(age_colors, ages):
                    ax.plot(unforgiveable_errors[age].T, color=c, alpha=0.1)
                    errorfill(range(m[age].shape[0]), m[age], se[age], color=c, ax=ax)
                ax.set_xlabel("Trial blocks")
                ax.set_ylabel(ylabel)
        return perseverative_errors, unforgiveable_errors

    def compare_trial_count(self, session_type):
        """
        Plot number of trials in young versus aged in this session.

        """
        trials = {
            age: [
                self.data[mouse][session_type].behavior.data["ntrials"]
                for mouse in self.meta["grouped_mice"][age]
            ]
            for age in ages
        }

        self.scatter_box(trials, "Trials")

        return trials

    ############################ ACTIVITY FUNCTIONS ##########################

    def percent_fading_neurons(self, mouse, session_type, x='trial',
                               z_threshold=None, x_bin_size=6, alpha=0.01):
        trends, binned_activations, slopes = \
            self.find_activity_trends(mouse, session_type, x=x,
                                      z_threshold=z_threshold, x_bin_size=x_bin_size,
                                      alpha=alpha, data_type='S')
        n_fading = len(trends['decreasing'])
        n_neurons = self.data[mouse][session_type].imaging['n_neurons']
        p_fading = n_fading / n_neurons

        return p_fading

    def plot_percent_fading_neurons(self, session_pair=('Goals4','Reversal'), x='trial',
                                    z_threshold=None, x_bin_size=6, alpha=0.01):
        """
        Plot the percentage of fading neurons on two sessions, usually Goals4 and Reversal.

        :return
        ---
        df: DataFrame
            'age': 'young' or 'aged'
            'mouse': mouse name
            'p_fading': percentage of fading neurons
            'session': session
        """
        data = {'age': [],
                'mouse': [],
                'p_fading': [],
                'session': []}
        for mouse in self.meta['mice']:
            age = 'aged' if mouse in aged_mice else 'young'

            for session in session_pair:
                p_fading = self.percent_fading_neurons(mouse, session, x=x, z_threshold=z_threshold,
                                                       x_bin_size=x_bin_size, alpha=alpha)

                for key, value in zip(data.keys(), [age, mouse, p_fading, session]):
                    data[key].append(value)

        df = pd.DataFrame(data)

        fig, axs = plt.subplots(1,2, sharey=True)
        fig.subplots_adjust(wspace=0)
        for age, color, ax in zip(ages, age_colors, axs):
            plot_me = [df['p_fading'].loc[np.logical_and(df['session']==session, df['age']==age)]
                       for session in session_pair]

            ax.plot(session_pair, plot_me, c=color)
            ax.set_title(age)
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_xlim([-0.5, 1.5])
        axs[0].set_ylabel('Percent fading neurons')
        fig.tight_layout()

        return df

    ############################ OVERLAP FUNCTIONS ############################

    def find_all_overlaps(self, show_plot=True):
        """
        Find average overlap with all sessions referenced to every other session, plotted as a line plot.

        """
        n_sessions = len(self.meta["session_types"])
        overlaps = nan_array((len(self.meta["mice"]), n_sessions, n_sessions))

        for i, mouse in enumerate(self.meta["mice"]):
            overlaps[i] = self.get_all_overlaps_mouse(mouse, show_plot=False)

        if show_plot:
            fig, ax = plt.subplots()
            m = np.mean(overlaps, axis=0)
            se = sem(overlaps, axis=0)
            x = self.meta["session_labels"]
            for y, yerr in zip(m, se):
                errorfill(x, y, yerr, ax=ax)

            ax.set_ylabel("Proportion of registered cells")
            plt.setp(ax.get_xticklabels(), rotation=45)

            return overlaps, ax

        return overlaps

    def get_all_overlaps_mouse(self, mouse, show_plot=True):
        """
        Get overlaps for all session pairs for one mouse and plot as a matrix.

        """
        n_sessions = len(self.meta["session_types"])
        r, c = np.indices((n_sessions, n_sessions))
        overlaps = np.ones((n_sessions, n_sessions))
        for session_pair, i, j in zip(
            product(self.meta["session_types"], self.meta["session_types"]),
            r.flatten(),
            c.flatten(),
        ):
            if i != j:
                overlaps[i, j] = self.find_overlap_session_pair(mouse, session_pair)

        if show_plot:
            fig, ax = plt.subplots()
            ax.imshow(overlaps)

        return overlaps

    def find_overlap_session_pair(self, mouse, session_pair):
        """
        Find overlap in one mouse for a session pair.

        """
        overlap_map = self.get_cellreg_mappings(
            mouse, session_pair, detected="first_day"
        )[0]

        n_neurons_reactivated = np.sum(overlap_map.iloc[:, 1] != -9999)
        n_neurons_first_day = len(overlap_map)

        overlap = n_neurons_reactivated / n_neurons_first_day

        return overlap

    def find_overlaps(self, session_pair):
        """
        Find all overlaps for a session pair.

        """
        overlaps = {
            mouse: self.find_overlap_session_pair(mouse, session_pair)
            for mouse in self.meta["mice"]
        }

        return overlaps

    def plot_overlaps(self, session_pair1, session_pair2):
        """
        Plots the proportion overlap of detected neurons between a session pair
        between two session pairs. Basically compares the overlap in one
        session pair to the overlap in another session pair.

        :parameters
        ---
        session_pair1, session_pair2: tuples
            Two session pairs that you want to compare.

        :returns
        ---
        overlaps: list of dicts
            [session_pair1 dict {mouse: overlap proportion},
             session_pair2 dict {mouse: overlap proportion},
             ]

        overlaps_grouped: dict
            {age: np.array(mouse, session_pair)}

        """
        overlaps = [
            self.find_overlaps(session_pair)
            for session_pair in [session_pair1, session_pair2]
        ]
        xlabels = [
            f"{session_pair[0]} x \n {session_pair[1]}"
            for session_pair in [session_pair1, session_pair2]
        ]
        fig, ax = plt.subplots(figsize=(3, 6))
        overlaps_grouped = dict()
        for color, age in zip(age_colors, ages):
            overlaps_grouped[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                overlaps_this_mouse = [overlap[mouse] for overlap in overlaps]
                ax.plot(xlabels, overlaps_this_mouse, color=color)

                overlaps_grouped[age].append(overlaps_this_mouse)

            overlaps_grouped[age] = np.vstack(overlaps_grouped[age])

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        ax.set_ylabel("Overlap [proportion]")
        fig.tight_layout()

        return overlaps, overlaps_grouped

    ############################ PLACE FIELD FUNCTIONS ############################
    def get_placefields(self, mouse, session_type, nbins=125, velocity_threshold=7):
        """
        Get place fields from one mouse for a session. If the specified parameters do not match the ones
        from PlaceFields(), recompute them.

        """
        session = self.data[mouse][session_type]
        existing_pfs = session.spatial.data["placefields"]
        if (
            existing_pfs.shape[1] == nbins
            and session.spatial.meta["velocity_threshold"] == velocity_threshold
        ):
            pf = existing_pfs
        else:
            PF = PlaceFields(
                np.asarray(session.behavior.data["df"]["t"]),
                np.asarray(session.behavior.data["df"]["x"]),
                np.zeros_like(session.behavior.data["df"]["x"]),
                session.imaging["S"],
                bin_size=None,
                nbins=nbins,
                circular=False,
                linearized=True,
                fps=session.behavior.meta["fps"],
                shuffle_test=False,
                velocity_threshold=velocity_threshold,
            )
            pf = PF.data["placefields"]

        return pf

    def PV_corr_pair(self, mouse, session_pair, nbins=125,
                velocity_threshold=7, corr='spearman'):
        """
        Compute the PV correlation from a mouse for a session pair.

        """
        # Get placefields, make new ones if the specified parameters
        # don't match existing ones.
        pfs = {session:
            self.get_placefields(mouse, session, nbins=nbins,
                                 velocity_threshold=velocity_threshold)
            for session in session_pair
        }

        # Get correlation function.
        corr_fun = spearmanr if corr == "spearman" else pearsonr

        # Register neurons and place fields.
        trimmed_map = np.asarray(
            self.get_cellreg_mappings(mouse, session_pair, detected="everyday")[
                0
            ]
        )
        s1, s2 = [
            pfs[session][neurons].T
            for neurons, session in zip(trimmed_map.T, session_pair)
        ]

        # Get correlations.
        rhos = []
        for x, y in zip(s1, s2):
            rhos.append(corr_fun(x, y, nan_policy="omit")[0])

        return rhos, pfs

    def compare_PV_corrs_by_bin(self, mouse, session_pair1, session_pair2,
                                nbins=125, velocity_threshold=7,
                                corr='spearman', colors=['darkgray', 'steelblue'],
                                show_plot=True):
        """
        Plot PV correlation values from one mouse from two session pairs. This allows comparison of
        PV correlations e.g. between Training3 x Training4 and Training4 x Reversal.

        """
        if nbins != 125:
            warnings.warn('Plotting reward site only works for nbins==125', UserWarning)

        session_pairs = [session_pair1, session_pair2]
        behavior_data = [self.data[mouse][session_pair[1]].behavior.data
                         for session_pair in session_pairs]

        port_bins = [
            find_reward_spatial_bins(
                data['df']['lin_position'],
                np.asarray(data['lin_ports']),
                spatial_bin_size_radians=0.05,
            )[0]
            for data in behavior_data
        ]

        rhos = [self.PV_corr_pair(mouse, session_pair, nbins=nbins,
                                  velocity_threshold=velocity_threshold,
                                  corr=corr)[0]
                for session_pair in session_pairs]

        if show_plot:
            fig, ax = plt.subplots()
            for rho, color, reward_locations, behavior\
                    in zip(rhos, colors, port_bins, behavior_data):
                ax.plot(rho, color=color)

                [ax.axvline(x=reward_location, color=color)
                 for reward_location in reward_locations[behavior['rewarded_ports']]]

                ax.set_xlabel('Spatial location')
                ax.set_ylabel('Correlation coefficient')

        return rhos, port_bins

    def compare_reward_PV_corrs(self, mouse, spatial_bin_window=5,
                                nbins=125, velocity_threshold=0,
                                corr='spearman',
                                colors=['darkgray', 'steelblue']):
        """
        Compare mean PV correlations of the bins around reward sites for two session pairs.

        :parameters
        ---
        mouse: str
            Mouse name.

        spatial_bin_window: int
            Number of spatial bins to take the mean across, flanking the reward site.

        """
        session_pairs = [('Goals3', 'Goals4'), ('Goals4', 'Reversal')]
        goals = {session: self.data[mouse][session].behavior.data['rewarded_ports']
                 for session in ['Goals4', 'Reversal']}

        binned_rhos, port_bins = \
            self.compare_PV_corrs_by_bin(mouse,
                                         session_pairs[0],
                                         session_pairs[1],
                                         nbins=nbins,
                                         velocity_threshold=velocity_threshold,
                                         corr=corr, show_plot=False)

        # Need to handle circularity of slices here.
        rhos = dict()
        never_rewarded = ~(goals['Goals4'] + goals["Reversal"])
        for rhos_pair, ports, session_pair\
                in zip(binned_rhos, port_bins, session_pairs):
            rhos[session_pair] = dict()
            current_goals = goals[session_pair[1]]

            rhos_at_rewards = [list(islice(cycle(rhos_pair), reward-spatial_bin_window,
                                           reward+spatial_bin_window)) for reward in ports[current_goals]]

            rhos_at_nonrewards = [list(islice(cycle(rhos_pair), reward-spatial_bin_window,
                                              reward+spatial_bin_window)) for reward in ports[never_rewarded]]

            rhos[session_pair]['currently_rewarded'] = np.nanmean([np.nanmean(r) for r in rhos_at_rewards])
            rhos[session_pair]['never_rewarded'] = np.nanmean([np.nanmean(r) for r in rhos_at_nonrewards])

            if session_pair[1] == 'Reversal':
                rhos_at_previous = [list(islice(cycle(rhos_pair), reward-spatial_bin_window,
                                                reward+spatial_bin_window)) for reward in ports[goals['Goals4']]]
                rhos[session_pair]['previously_rewarded'] = np.nanmean([np.nanmean(r) for r in rhos_at_previous])

        return rhos

    def plot_reward_PV_corrs_v1(self, mice):
        """
        Plot PV correlations around reward locations for each mouse, split into session pairs for each subplot,
        then into currently/never rewarded within each subplot.

        :param mice:
        :return:
        """
        session_pairs = [('Goals3', 'Goals4'), ('Goals4', 'Reversal')]
        rhos = {mouse: self.compare_reward_PV_corrs(mouse) for mouse in mice}
        xlabels = [f'{session_pair[0].replace("Goals", "Training")} x {session_pair[1].replace("Goals", "Training")}'
                   for session_pair in session_pairs]

        fig, axs = plt.subplots(1,2,sharey=True)
        fig.subplots_adjust(wspace=0)

        for session_pair, xlabel, ax in zip(session_pairs, xlabels, axs):
            reward_types = rhos[mice[0]][session_pair].keys()
            xticks = [str(reward_str).replace('_', ' \n') for reward_str in reward_types]

            data = []
            for reward_type in reward_types:
                data.append([rhos[mouse][session_pair][reward_type] for mouse in mice])

            ax.boxplot(data)
            ax.set_xticklabels(xticks, rotation=45)
            ax.set_xlabel(xlabel)

        axs[0].set_ylabel('PV correlation coefficient')
        fig.tight_layout()

        return rhos

    def plot_reward_PV_corrs_v2(self, mice):
        """
        Plot PV correlations around reward locations for each mouse, split into currently/never/previously
        rewarded for each subplot, then into session pair  within each subplot.

        :param mice:
        :return:
        """
        session_pairs = [('Goals3', 'Goals4'), ('Goals4', 'Reversal')]
        rhos = {mouse: self.compare_reward_PV_corrs(mouse) for mouse in mice}
        xticks = [f'{session_pair[0].replace("Goals", "Training")} x \n{session_pair[1].replace("Goals", "Training")}'
                   for session_pair in session_pairs]

        fig, axs = plt.subplots(1,3,sharey=True)
        fig.subplots_adjust(wspace=0)

        for ax, reward_type in zip(axs[:-1], ['currently_rewarded', 'never_rewarded']):
            data = []
            for session_pair in session_pairs:
                data.append([rhos[mouse][session_pair][reward_type] for mouse in mice])

            ax.boxplot(data)
            ax.set_xticklabels(xticks, rotation=45)
            ax.set_xlabel(reward_type.replace('_', ' \n'))

        axs[-1].boxplot([rhos[mouse][session_pairs[1]]['previously_rewarded'] for mouse in mice])
        axs[-1].set_xticklabels([xticks[-1]], rotation=45)
        axs[-1].set_xlabel('previously \nrewarded')
        axs[0].set_ylabel('PV correlation coefficient')

        fig.tight_layout()

        return rhos

    def get_split_trial_pfs(self, mouse, session_type, nbins=125, show_plot=False):
        """
        Make place fields separately for even and odd trials.

        :return
        ---
        split_pfs: dict
            Place fields separated into even versus odd trials.

        """
        session = self.data[mouse][session_type]
        existing_rasters = session.spatial.data["rasters"]

        if existing_rasters.shape[2] == nbins:
            rasters = existing_rasters
        else:
            rasters = session.spatial_activity_by_trial(nbins)[0]

        split_rasters = {"even": rasters[:, ::2, :], "odd": rasters[:, 1::2, :]}
        split_pfs = {
            trial_type: np.nanmean(split_rasters[trial_type], axis=1)
            for trial_type in ["even", "odd"]
        }

        if show_plot:
            order = np.argsort(np.argmax(split_pfs["even"], axis=1))
            fig, axs = plt.subplots(1, 2, figsize=(8, 6))
            for ax, trial_type in zip(axs, ["even", "odd"]):
                ax.imshow(split_pfs[trial_type][order], aspect="auto")
                ax.set_title(f"{trial_type} trials")
            fig.supylabel("Neuron #")
            fig.supxlabel("Linearized position")

            return split_pfs

    def session_pairwise_PV_corr_efficient(
        self, mouse, nbins=125, corr="spearman", show_plot=False
    ):
        """
        Efficiently compute PV correlations for each session pair for a single mouse.
        THe efficiency comes from loading the place fields all at once for one mouse as opposed to
        loading the data for each mouse for each session pair each time.

        :param mouse:
        :param nbins:
        :param corr:
        :param show_plot:
        :return:
        """
        pfs = {}
        for session in self.meta["session_types"]:
            pfs[session] = self.get_placefields(mouse, session, nbins=nbins)

        corr_fun = spearmanr if corr == "spearman" else pearsonr

        shape = (len(self.meta["session_types"]), len(self.meta["session_types"]))
        corr_matrix = nan_array(shape)
        for i, session_pair in enumerate(product(self.meta["session_types"], repeat=2)):
            same_session = session_pair[0] == session_pair[1]
            row, col = np.unravel_index(i, shape)

            if same_session:
                split_pfs = self.get_split_trial_pfs(
                    mouse, session_pair[0], nbins=nbins
                )
                even, odd = [split_pfs[trial_type].T for trial_type in ["even", "odd"]]

                rhos = []
                for x, y in zip(even, odd):
                    rhos.append(corr_fun(x, y, nan_policy="omit")[0])

            else:
                trimmed_map = np.asarray(
                    self.get_cellreg_mappings(mouse, session_pair, detected="everyday")[
                        0
                    ]
                )
                s1, s2 = [
                    pfs[session][neurons].T
                    for neurons, session in zip(trimmed_map.T, session_pair)
                ]

                rhos = []
                for x, y in zip(s1, s2):
                    rhos.append(corr_fun(x, y, nan_policy="omit")[0])

            corr_matrix[row, col] = np.nanmean(rhos)

        if show_plot:
            fig, ax = plt.subplots()
            ax.imshow(corr_matrix)

        return corr_matrix

    def PV_corr_all_mice(self, nbins=30):
        """
        Plot the average PV correlations for each session pair for every mouse.
        Takes a long time.

        :return
        ---
        corr_matrices: dict
            PV correlation matrix for each mouse.
        """
        corr_matrices = dict()
        for mouse in self.meta["mice"]:
            print(f"Analyzing {mouse}...")
            corr_matrices[mouse] = self.session_pairwise_PV_corr_efficient(
                mouse, nbins=nbins
            )

        return corr_matrices

    def plot_corr_matrix(self, corr_matrices, ages_to_plot=ages):
        """
        Plot correlation matrices, averaged across all mice.

        """
        fig, axs = plt.subplots(1, len(ages_to_plot), figsize=(12, 5))
        both_ages = True
        try:
            len(axs)
        except:
            both_ages = False
            axs = [axs]

        matrices = []
        for ax, age in zip(axs, ages_to_plot):
            matrix = np.nanmean(
                [corr_matrices[mouse] for mouse in self.meta["grouped_mice"][age]],
                axis=0,
            )

            matrices.append(matrix)
            im = ax.imshow(matrix)
            ax.set_title(f"{age}")
            ax.set_xlabel("Day")
            ax.set_ylabel("Day")
            ax.set_xticks(range(len(self.meta["session_types"])))
            ax.set_yticks(range(len(self.meta["session_types"])))
            ax.set_xticklabels(self.meta["session_labels"], rotation=90)
            ax.set_yticklabels(self.meta["session_labels"])
        fig.tight_layout()
        min_clim = np.min(matrices)
        max_clim = np.max(matrices)

        if both_ages:
            for ax in axs.flatten():
                for im in ax.get_images():
                    im.set_clim(min_clim, max_clim)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Mean PV correlation [Spearman rho]")

    def get_diagonals(self, corr_matrices):
        """
        Get diagonals for each mouse's PV correlation matrix. This is useful for looking at correlation
        coefficients for each day lag.

        """
        data = {}
        for mouse in self.meta["mice"]:
            data[mouse] = {
                "coefs": [],
                "day_lag": [],
            }
            for i in range(len(self.meta["session_types"])):
                rhos = np.diag(corr_matrices[mouse], k=i)

                data[mouse]["coefs"].extend(rhos)
                data[mouse]["day_lag"].extend(np.ones_like(rhos) * i)

            data[mouse]["coefs"] = np.asarray(data[mouse]["coefs"])
            data[mouse]["day_lag"] = np.asarray(data[mouse]["day_lag"])

        return data

    def compare_PV_corrs(self, corr_matrices):
        """
        For each day lag, plot PV correlations of young versus aged mice.

        """
        data = self.get_diagonals(corr_matrices)
        n_sessions = len(self.meta["session_types"])
        PV_corrs = {}
        for day_lag in range(n_sessions):
            PV_corrs[day_lag] = dict()
            for age in ages:
                PV_corrs[day_lag][age] = []
                for mouse in self.meta["grouped_mice"][age]:
                    coefs = data[mouse]["coefs"][data[mouse]["day_lag"] == day_lag]
                    coefs = coefs[~np.isnan(coefs)]
                    PV_corrs[day_lag][age].extend(coefs)

        fig, axs = plt.subplots(1, n_sessions, sharey=True, figsize=(10.5, 5))
        for day_lag, ax in enumerate(axs):
            self.scatter_box(PV_corrs[day_lag], ax=ax)
            ax.set_title(f"{day_lag} days apart")
        axs[0].set_ylabel("PV correlation [Spearman rho]")
        fig.tight_layout()
        self.set_age_legend(fig)

        return PV_corrs

    def plot_session_PV_corr_comparisons(
        self,
        corr_matrices,
        session_pairs=(("Goals3", "Goals4"), ("Goals4", "Reversal")),
    ):
        """
        For the specified sessions, plot the PV correlation coefficient between those sessions,
        separated by age.


        """
        data = {session_pair: dict() for session_pair in session_pairs}
        for session_pair in session_pairs:
            s1 = self.meta["session_types"].index(session_pair[0])
            s2 = self.meta["session_types"].index(session_pair[1])
            data[session_pair] = dict()
            for age in ages:
                data[session_pair][age] = []
                for mouse in self.meta["grouped_mice"][age]:
                    data[session_pair][age].append(corr_matrices[mouse][s1, s2])

        fig, axs = plt.subplots(1, len(session_pairs), sharey=True)
        for ax, age, color in zip(axs, ages, age_colors):
            boxes = ax.boxplot(
                [data[session_pair][age] for session_pair in session_pairs],
                widths=0.75,
                showfliers=False,
                zorder=0,
                patch_artist=True,
            )

            [
                ax.scatter(
                    jitter_x(np.ones_like(data[session_pair][age]) * (i + 1), 0.05),
                    data[session_pair][age],
                    color=color,
                    edgecolor="k",
                    zorder=1,
                    s=50,
                )
                for i, session_pair in enumerate(session_pairs)
            ]

            for patch, med in zip(boxes["boxes"], boxes["medians"]):
                patch.set_facecolor(color)
                med.set(color="k")
            ax.set_xticklabels(
                [
                    f'{session_pair[0].replace("Goals", "Training")} vs. \n'
                    f'{session_pair[1].replace("Goals", "Training")}'
                    for session_pair in session_pairs
                ],
                rotation=45,
            )
            ax.set_title(age)
        axs[0].set_ylabel("Spatial PV correlation coefficients")
        fig.tight_layout()

        df = pd.concat([pd.concat({k: pd.Series(v) for k,v in data[pair].items()}
                                  ) for pair in session_pairs]
                       , axis=1, keys=session_pairs)

        return data, df

    def get_drift_rate(self, corr_matrices):
        """
        Compute the drift rate for each mouse by doing a spearman correlation of PV correlation
        coefficient against day lag.

        :return
        ---
        drift_rates: dict
            Drift rate for each mouse.
        """
        data = self.get_diagonals(corr_matrices)
        drift_rates = {
            mouse: spearmanr(
                data[mouse]["day_lag"], data[mouse]["coefs"], nan_policy="omit"
            )[0]
            for mouse in self.meta["mice"]
        }

        return drift_rates

    def compare_drift_rates(self, corr_matrices, show_plot=True):
        """
        Plot the drift rates of aged versus young mice.

        :param corr_matrices:
        :param show_plot:
        :return:
        """
        drift_rates = self.get_drift_rate(corr_matrices)
        drift_rate_ages = {
            age: [drift_rates[mouse] for mouse in self.meta["grouped_mice"][age]]
            for age in ages
        }

        if show_plot:
            self.scatter_box(
                drift_rate_ages, ylabel="Drift rates " "[more negative = more drift]"
            )

        return drift_rate_ages

    def plot_one_drift_rate(self, mouse, corr_matrices):
        """
        Plot the PV correlations against day lag for one mouse.

        """
        data = self.get_diagonals(corr_matrices)
        fig, ax = plt.subplots()
        ax.scatter(data[mouse]["day_lag"], data[mouse]["coefs"])
        ax.set_xticks(range(len(self.meta["session_types"])))
        ax.set_xlabel("Day lag")
        ax.set_ylabel("PV correlation [rho]")

    def corr_PV_corr_to_behavior(self, corr_matrices, performance_metric):
        """
        Correlate the PV correlation of Training4 x Reversal against the performance metric during Reversal.

        :returns
        ---
        r: float
            Spearman correlation coefficient.

        p: float
            P-value of the correlation.
        """
        performance = self.plot_all_behavior(
            window=None,
            strides=None,
            show_plot=False,
            performance_metric=performance_metric,
        )[2]
        PV_corrs = {
            mouse: np.diag(corr_matrices[mouse], k=4)[0] for mouse in self.meta["mice"]
        }
        PV_corrs_grouped = {
            age: [
                np.diag(corr_matrices[mouse], k=4)[0]
                for mouse in self.meta["grouped_mice"][age]
            ]
            for age in ages
        }
        PV_corrs = [PV_corrs[mouse] for mouse in self.meta["mice"]]

        reversal_performances = performance[performance["session_types"] == "Reversal"]
        performances = reversal_performances["metric"].tolist()
        performances_grouped = {
            age: [
                reversal_performances[reversal_performances["mice"] == mouse][
                    "metric"
                ].values[0]
                for mouse in self.meta["grouped_mice"][age]
            ]
            for age in ages
        }

        fig, ax = plt.subplots()
        ylabels = {"hits": "Hit rate", "CRs": "Correct rejection rate", "d_prime": "d'"}
        for age, color in zip(ages, age_colors):
            ax.scatter(PV_corrs_grouped[age], performances_grouped[age], color=color)
            ax.set_ylabel(ylabels[performance_metric])
            ax.set_xlabel("PV correlation [Spearman rho]")

        r, pvalue = spearmanr(PV_corrs, performances)

        return r, pvalue

    def map_placefields(self, mouse, session_types, neurons_from_session1=None):
        """
        Register place fields across sessions.

        """
        # Get neurons and cell registration mappings.
        trimmed_map, global_idx = self.get_cellreg_mappings(
            mouse, session_types, neurons_from_session1=neurons_from_session1
        )[:-1]

        # Get fields.
        fields = {
            session_type: self.data[mouse][session_type].spatial.data[
                "placefields_normalized"
            ]
            for session_type in session_types
        }

        # Prune fields to only take the neurons that were mapped.
        fields_mapped = []
        for i, session_type in enumerate(session_types):
            mapped_neurons = trimmed_map.iloc[trimmed_map.index.isin(global_idx), i]
            fields_mapped.append(fields[session_type][mapped_neurons])

        return fields_mapped

    def correlate_fields(
        self, mouse, session_types, neurons_from_session1=None, show_histogram=True
    ):
        """
        Correlate place fields for one mouse across sessions.

        """
        fields_mapped = self.map_placefields(
            mouse, session_types, neurons_from_session1=neurons_from_session1
        )

        # Do correlation for each neuron and store r and p value.
        corrs = {"r": [], "p": []}
        for field1, field2 in zip(*fields_mapped):
            r, p = spearmanr(field1, field2, nan_policy="omit")
            corrs["r"].append(r)
            corrs["p"].append(p)

        if show_histogram:
            plt.hist(corrs["r"])

        return corrs

    def plot_aged_pf_correlation(
        self, session_types=("Goals3", "Goals4"), show_plot=True, place_cells=True
    ):
        """
        Make violin plots for place field correlations for each mouse. Note that this is NOT a PV correlation.
        Instead of correlating N-length vectors for each spatial bin, correlate the fields for each cell.

        """
        r_values = dict()
        for age in ages:
            r_values[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                if place_cells:
                    neurons = self.data[mouse][session_types[0]].spatial.data[
                        "place_cells"
                    ]
                else:
                    neurons = None

                rs_this_mouse = np.asarray(
                    self.correlate_fields(
                        mouse,
                        session_types,
                        show_histogram=False,
                        neurons_from_session1=neurons,
                    )["r"]
                )
                rs = rs_this_mouse[~np.isnan(rs_this_mouse)]

                r_values[age].append(rs)

        if show_plot:
            fig, axs = plt.subplots(1, 2, sharey=True)
            fig.subplots_adjust(wspace=0)

            for ax, age in zip(axs, ages):
                ax.violinplot(r_values[age], showmedians=True, showextrema=False)
                ax.set_title(age)
                ax.set_xticks(np.arange(1, len(self.meta["grouped_mice"][age]) + 1))
                ax.set_xticklabels(self.meta["grouped_mice"][age], rotation=45)

                if age == "young":
                    ax.set_ylabel("Place field correlations [r]")

        return r_values

    def plot_aged_pf_correlation_comparisons(
        self,
        session_pair1=("Goals3", "Goals4"),
        session_pair2=("Goals4", "Reversal"),
        place_cells=False,
    ):
        """
        Make boxplots for place field correlations across one session pair and compare it with another session pair.

        """
        r_values = dict()
        for session_pair in [session_pair1, session_pair2]:
            r_values[session_pair] = self.plot_aged_pf_correlation(
                session_pair, show_plot=False, place_cells=place_cells
            )

        fig, axs = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0)
        for age, ax, color in zip(ages, axs, age_colors):
            mice = self.meta["grouped_mice"][age]
            positions = [
                np.arange(start, start + 3 * len(mice), 3) for start in [-0.5, 0.5]
            ]
            label_positions = np.arange(0, 3 * len(mice), 3)

            for session_pair, position in zip(
                [session_pair1, session_pair2], positions
            ):
                boxes = ax.boxplot(
                    r_values[session_pair][age],
                    positions=position,
                    patch_artist=True,
                )
                for patch, med in zip(boxes["boxes"], boxes["medians"]):
                    patch.set_facecolor(color)
                    med.set(color="k")

            if age == "aged":
                ax.set_yticks([])
            else:
                ax.set_ylabel("Spatial correlations [Spearman rho]")
            ax.set_xticks(label_positions)
            ax.set_xticklabels(mice, rotation=45)

        return r_values

    def scrollplot_rasters_by_day(
        self, mouse, session_types, neurons_from_session1=None, mode="scroll"
    ):
        """
        Visualize binned spatial activity by each trial across
        multiple days.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_types: array-like of strs
            Must correspond to at least two of the sessions in the
            session_types class attribute (e.g. 'CircleTrackGoals2'.

        neurons_from_session1: array-like of scalars
            Neuron indices to include from the first session in
            session_types.

        """
        # Get neuron mappings.
        sessions = self.data[mouse]
        trimmed_map, global_idx = self.get_cellreg_mappings(
            mouse, session_types, neurons_from_session1=neurons_from_session1
        )[:-1]

        # Gather neurons and build dictionaries for HoloMap.
        daily_rasters = []
        placefields = []
        ax_titles = [title.replace("Goals", "Training") for title in session_types]
        for i, session_type in enumerate(session_types):
            neurons_to_analyze = trimmed_map.iloc[trimmed_map.index.isin(global_idx), i]

            if mode == "holoviews":
                rasters = sessions[session_type].viz_spatial_trial_activity(
                    neurons=neurons_to_analyze, preserve_neuron_idx=False
                )
            elif mode in ["png", "scroll"]:
                rasters = (
                    sessions[session_type].spatial.data["rasters"][neurons_to_analyze]
                    > 0
                )

                placefields.append(
                    sessions[session_type].spatial.data["placefields_normalized"][
                        neurons_to_analyze
                    ]
                )
            else:
                raise ValueError("mode must be holoviews, png, or scroll")
            daily_rasters.append(rasters)

        if mode == "png":
            for neuron in range(daily_rasters[0].shape[0]):
                fig, axs = plt.subplots(1, len(daily_rasters))
                for s, ax in enumerate(axs):
                    ax.imshow(daily_rasters[s][neuron], cmap="gray")

                fname = os.path.join(
                    r"Z:\Will\Drift\Data",
                    mouse,
                    r"Analysis\Place fields",
                    f"Neuron {neuron}.png",
                )
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)
        elif mode == "scroll":
            ScrollPlot(
                plot_daily_rasters,
                current_position=0,
                nrows=len(session_types),
                ncols=2,
                rasters=daily_rasters,
                tuning_curves=placefields,
                titles=ax_titles,
            )

        # List of dictionaries. Do HoloMap(daily_rasters) in a
        # jupyter notebook.
        return daily_rasters

    def plot_spatial_info(
        self,
        session_type,
        ax=None,
        show_plot=True,
        aggregate_mode="median",
        place_cells_only=False,
    ):
        """
        Plot spatial info for each mouse during one session, aggregated across cells.

        :parameters
        ---
        session_type: str
            Session type.

        aggregate_mode: str
            "median", "mean", or "all".

        """
        if ax is None:
            fig, ax = plt.subplots()
            ylabel = 'Spatial info [z]'
        else:
            ylabel = ''

        if aggregate_mode not in ["median", "mean", "all"]:
            raise ValueError("Invalid aggregate_mode")

        spatial_info = dict()
        for age in ages:
            spatial_info[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                SIs = self.data[mouse][session_type].spatial.data["spatial_info_z"]

                if place_cells_only:
                    SIs = SIs[
                        self.data[mouse][session_type].spatial.data["place_cells"]
                    ]

                if aggregate_mode == "median":
                    spatial_info[age].append(np.median(SIs))
                elif aggregate_mode == "mean":
                    spatial_info[age].append(np.mean(SIs))
                elif aggregate_mode == "all":
                    spatial_info[age].extend(SIs)

        if show_plot:
            self.scatter_box(spatial_info, ax=ax)
            ax.set_ylabel(ylabel)

        return spatial_info

    def plot_all_spatial_info(self, aggregate_mode="median", place_cells_only=False):
        """
        Plot aggregated spatial information for each mouse and each session.

        :returns
        ---
        spatial_info: dict
            Aggregated spatial information for each session, split into young and aged.

        df: DataFrame
            Aggregated spatial information in the form of a dataframe, but also with
            animal identifiers.

        """
        fig, axs = plt.subplots(1, len(self.meta["session_types"]), sharey=True)
        fig.subplots_adjust(wspace=0)

        spatial_infos = dict()
        for ax, session_type, title in zip(
            axs, self.meta["session_types"], self.meta["session_labels"]
        ):
            spatial_infos[session_type] = self.plot_spatial_info(
                session_type,
                ax=ax,
                aggregate_mode=aggregate_mode,
                place_cells_only=place_cells_only,
            )
            ax.set_xticks([])
            ax.set_title(title)

            if session_type == "Goals1":
                ylabel = f"Spatial information ({aggregate_mode})"
                if aggregate_mode in ["median", "mean"]:
                    ylabel += " per mouse"
                ylabel += " [z]"
                ax.set_ylabel(ylabel)

        # fig, ax = plt.subplots()
        # for age, age_color in zip(ages, age_colors):
        #     ax.plot(self.meta['session_labels'],
        #             [median_spatial_infos[session_type][age]
        #              for session_type in self.meta['session_types']],
        #             color=age_color)

        infos = []
        ages_long = []
        sessions_long = []
        mice = []
        for age in ages:
            for session_type in self.meta["session_types"]:
                data = spatial_infos[session_type][age]
                infos.extend(data)
                ages_long.extend([age for i in range(len(data))])
                sessions_long.extend([session_type for i in range(len(data))])
                mice.extend(self.meta["grouped_mice"][age])

        df = pd.DataFrame(
            {
                "spatial_info": infos,
                "age": ages_long,
                "session_types": sessions_long,
                "mouse": mice,
            }
        )

        return spatial_infos, df

    def spatial_info_anova(self, aggregate_mode="median", place_cells_only=False):
        """
        Do ANOVA on the spatial information.

        :returns
        ---
        anova_df: DataFrame
            Results of the ANOVA.

        pairwise_df: DataFrame
            Results of the post-hoc pairwise t-tests.

        df: DataFrame
            Spatial information.
        """
        spatial_infos, df = self.plot_all_spatial_info(
            aggregate_mode=aggregate_mode, place_cells_only=place_cells_only
        )

        anova_df = pg.mixed_anova(
            df,
            dv="spatial_info",
            within="session_types",
            between="age",
            subject="mouse",
        )
        pairwise_df = df.pairwise_ttests(
            dv="spatial_info", between=["session_types", "age"], padjust="fdr_bh"
        )

        return anova_df, pairwise_df, df

    def plot_reliabilities(self, session_type, field_threshold=0.15, show_plot=True):
        """
        Plot the reliability for each mouse, split by young and aged. Reliability is defined as the fraction
        of trials where there was a calcium transient inside the field. The field is every spatial bin where
        there is average activity that exceeds the field_threshold * the peak average activity.

        :parameters
        ---
        session_type: str
            Session type.

        field_threshold: float
            Percentage of the peak activity that defines the extent of the field.

        """
        reliabilities = {}
        for age in ages:
            reliabilities[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                session = self.data[mouse][session_type]
                place_cells = session.spatial.data["place_cells"]
                reliabilities_ = [
                    session.placefield_reliability(
                        neuron,
                        field_threshold=field_threshold,
                        even_split=True,
                        split=1,
                        show_plot=False,
                    )
                    for neuron in place_cells
                ]
                reliabilities[age].append(reliabilities_)

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, color, ax in zip(ages, age_colors, axs):
                mice = self.meta["grouped_mice"][age]
                boxes = ax.boxplot(reliabilities[age], patch_artist=True)
                for patch, med in zip(boxes["boxes"], boxes["medians"]):
                    patch.set_facecolor(color)
                    med.set(color="k")

                if age == "aged":
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("Place cell reliability")
                ax.set_xticklabels(mice, rotation=45)

        return reliabilities

    def plot_reliabilities_comparisons(
        self, session_types=("Goals4", "Reversal"), field_threshold=0.5, show_plot=True
    ):
        """
        Plot the distribution of reliabilities for each cell in each mouse, separated by age, for the two sessions.
        Also plot the mean field reliability separated by age for the two sessions.

        :returns
        ---
        reliabilities: dict
            Place field reliabilities, first split by session, then by age. Each entry is a list containing
            all the reliabilities for each cell.

        mean_reliabilities: dict
            Same as above, but just the mean across cells for each mouse.

        """
        reliabilities = {}
        mean_reliabilities = {session_type: dict() for session_type in session_types}
        for session_type in session_types:
            reliabilities[session_type] = self.plot_reliabilities(
                session_type, field_threshold=field_threshold, show_plot=False
            )
            for age in ages:
                mean_reliabilities[session_type][age] = [
                    np.nanmean(r) for r in reliabilities[session_type][age]
                ]

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, ax, color in zip(ages, axs, age_colors):
                mice = self.meta["grouped_mice"][age]
                positions = [
                    np.arange(start, start + 3 * len(mice), 3) for start in [-0.5, 0.5]
                ]
                label_positions = np.arange(0, 3 * len(mice), 3)

                for session_type, position in zip(session_types, positions):
                    boxes = ax.boxplot(
                        reliabilities[session_type][age],
                        positions=position,
                        patch_artist=True,
                    )
                    for patch, med in zip(boxes["boxes"], boxes["medians"]):
                        patch.set_facecolor(color)
                        med.set(color="k")

                if age == "aged":
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("Place cell reliabilities")
                ax.set_xticks(label_positions)
                ax.set_xticklabels(mice, rotation=45)

            fig, axs = plt.subplots(1, len(session_types), sharey=True)
            fig.subplots_adjust(wspace=0)

            for session_type, ax in zip(session_types, axs):
                ylabel = (
                    "Mean place field reliability"
                    if session_type == session_types[0]
                    else None
                )
                self.scatter_box(mean_reliabilities[session_type], ylabel=ylabel, ax=ax)
                ax.set_title(session_type)

        return reliabilities, mean_reliabilities

    def correlate_field_reliability_to_performance(
        self,
        performance_metric="d_prime",
        session_types=("Goals4", "Reversal"),
        field_threshold=0.15,
    ):
        """
        It's in the name. Spearman correlation.

        :returns
        ---
        r: float
            Correlation coefficient.

        pvalue: float
            P-value of correlation.
        """
        reliabilities, mean_reliabilities = self.plot_reliabilities_comparisons(
            session_types=session_types,
            field_threshold=field_threshold,
            show_plot=False,
        )

        performance = self.plot_performance_session_type(
            "Reversal",
            window=None,
            performance_metric=performance_metric,
            show_plot=False,
        )
        fig, ax = plt.subplots()
        for age, color in zip(ages, age_colors):
            ax.scatter(
                mean_reliabilities["Reversal"][age], performance[age], color=color
            )

        ylabel = {"CRs": "Correct rejection rate", "hits": "Hit rate", "d_prime": "d'"}
        ax.set_ylabel(ylabel[performance_metric])
        ax.set_xlabel("Place field reliability")

        r, pvalue = spearmanr(
            np.hstack([mean_reliabilities["Reversal"][age] for age in ages]),
            np.hstack([performance[age] for age in ages]),
        )

        return r, pvalue

    def snakeplot_placefields(
        self,
        mouse,
        session,
        order=None,
        neurons=None,
        ax=None,
        normalize=True,
        show_plot=True,
        show_reward_sites=True,
    ):
        """
        Make snake plot of place fields for all neurons in one mouse for a session. Also plots the reward location
        in green.

        :returns
        ---
        ax: AxesSubplot handle

        placefields: array
            Normalized and sorted placefields.

        order: array
            Indices of the neurons, sorted by the order in which they appear on the track.

        """
        spatial_bin_size_radians = self.data[mouse][session].spatial.meta['bin_size']
        behavior_data = self.data[mouse][session].behavior

        session = self.data[mouse][session].spatial.data
        port_locations = np.asarray(behavior_data.data['lin_ports'])[behavior_data.data['rewarded_ports']]
        if neurons is None:
            neurons = np.arange(0, session["n_neurons"])
        if order is None:
            order = np.argsort(session["placefield_centers"][neurons])

        placefields = session["placefields_normalized"][neurons][order]

        # Get reward locations
        reward_location_bins, bins = find_reward_spatial_bins(
            session['x'],
            port_locations,
            spatial_bin_size_radians=spatial_bin_size_radians,
        )

        if show_plot:
            if normalize:
                placefields = placefields / np.max(placefields, axis=1, keepdims=True)
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5.5))
            ax.imshow(placefields)

            if show_reward_sites:
                [ax.axvline(x=reward_location_bin, color='g') for reward_location_bin in reward_location_bins]

            ax.axis("tight")
            ax.set_ylabel("Neuron #")
            ax.set_xlabel("Location")
        else:
            ax = None

        return ax, placefields, order

    def snakeplot_matched_placefields(self, mouse, session_types, sort_by_session=None):
        """
        Snake plot place fields that have been matched across sessions.

        """
        trimmed_map = self.get_cellreg_mappings(
            mouse, session_types, detected="everyday", neurons_from_session1=None
        )[0]
        trimmed_map = np.asarray(trimmed_map)
        age = "aged" if mouse in self.meta["grouped_mice"]["aged"] else "young"

        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in session_types
        ]
        fig, axs = plt.subplots(1, len(session_types))
        ax, placefields, order = self.snakeplot_placefields(
            mouse,
            session_types[sort_by_session],
            neurons=trimmed_map[:, sort_by_session],
            show_plot=False,
        )

        for i, (ax, label, session_type) in enumerate(
            zip(axs, session_labels, session_types)
        ):
            self.snakeplot_placefields(
                mouse, session_type, neurons=trimmed_map[:, i], order=order, ax=ax
            )
            ax.set_title(label)
        fig.suptitle(f"{mouse}, {age}")
        fig.tight_layout()

    ############################ DECODER FUNCTIONS ############################
    def decode_place(
        self,
        mouse,
        training_and_test_sessions,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        show_plot=True,
        predictors="cells",
    ):
        """
        Use a naive Bayes decoder to classify spatial position.
        Train with data from one session and test with another.

        :parameters
        ---
        mouse: str
            Mouse name.

        training_session and test_session: strs
            Training and test sessions.

        classifier: BernoulliNB() or MultinomialNB()
            Naive Bayes classifier to use.

        decoder_time_bin_size: float
            Size of time bin in seconds.

        n_spatial_bins: int
            Number of spatial bins to divide circle track into.
            Default to 36 for each bin to be 10 degrees.

        show_plot: boolean
            Flag for showing plots.
        """
        # Get sessions and neural activity.
        sessions = [self.data[mouse][session] for session in training_and_test_sessions]
        if predictors == "cells":
            neural_data = self.rearrange_neurons(
                mouse, training_and_test_sessions, data_type="S_binary"
            )
        elif predictors == "ensembles":
            neural_data = []
            registered_ensembles = self.match_ensembles(
                mouse, training_and_test_sessions
            )
            poor_matches = registered_ensembles["poor_matches"]
            for session_activations in registered_ensembles["matched_activations"]:
                neural_data.append(
                    zscore(np.squeeze(session_activations[~poor_matches]), axis=0)
                )
        else:
            raise ValueError
        fps = 15

        running = [
            self.data[mouse][session].spatial.data["running"]
            for session in training_and_test_sessions
        ]

        # Separate neural data into training and test.
        X = {"train": neural_data[0][:, running[0]].T}
        X["test"] = neural_data[1].T

        # Separate spatially binned location into training and test.
        y = {
            "train": format_spatial_location_for_decoder(
                sessions[0].behavior.data["df"]["lin_position"].values[running[0]],
                n_spatial_bins=n_spatial_bins,
                time_bin_size=1 / fps,
                fps=fps,
                classifier=classifier,
            )
        }
        y["test"] = format_spatial_location_for_decoder(
            sessions[1].behavior.data["df"]["lin_position"].values,
            n_spatial_bins=n_spatial_bins,
            time_bin_size=1 / fps,
            fps=fps,
            classifier=classifier,
        )

        # Fit the classifier and test on test data.
        classifier.fit(X["train"], y["train"])
        y_predicted = classifier.predict(X["test"])

        # Plot real and predicted spatial location.
        if show_plot:
            fig, ax = plt.subplots()
            ax.plot(y["test"][running[1]], alpha=0.5)
            ax.plot(y_predicted[running[1]], alpha=0.5)

        return y_predicted, X, y, classifier, running

        # X_test = S_list[1].T
        # y_predicted = classifier.predict(X_test)
        # y = sessions[1].data['behavior'].behavior_df['lin_position'].values
        # y_real = np.digitize(y, bins)

        # plt.plot(y_real, alpha=0.2)
        # plt.plot(y_predicted, alpha=0.2)

    def plot_reversal_decoding_error(
        self,
        mouse,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        error_time_bin_size=300 * 15,
        axs=None,
        color="cornflowerblue",
        predictors="cells",
    ):
        """
        Plot the spatial decoding error for a mouse. Train on Goals3 and test on Goals4.
        Additionally, train on Goals4 and test on Reversal.

        :returns
        ---
        decoding_errors: dict
            "Training" and "Reversal" correspond to the two session pairs listed above, respectively.
            Each entry is a list of errors (in spatial bins) across time bins.
        """

        decoding_errors = dict()
        for key, session_pair in zip(
            ["Training", "Reversal"], [("Goals3", "Goals4"), ("Goals4", "Reversal")]
        ):

            decoding_errors[key] = self.find_decoding_error(
                mouse,
                session_pair,
                classifier=classifier,
                n_spatial_bins=n_spatial_bins,
                error_time_bin_size=error_time_bin_size,
                show_plot=False,
                predictors=predictors,
            )[0]
        time_bins = range(len(decoding_errors["Training"]))

        if axs is None:
            fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
        for errors, ax in zip(
            [decoding_errors["Training"], decoding_errors["Reversal"]],
            axs,
        ):
            means = [np.nanmean(error) for error in errors]
            # sems = [np.nanstd(error) / np.sqrt(error.shape[0]) for error in errors]
            ax.plot(time_bins, means, color=color, alpha=0.2)

        return decoding_errors

    def spatial_decoding_anova_binned(
        self,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        error_time_bin_size=300 * 15,
        predictors="cells",
    ):
        """
        Do ANOVA on spatial decoding error across ages.

        :returns
        ---
        decoding_errors: dict
            Average decoding error, binned by time for each mouse, then separated by ages, and then by session.

        df: DataFrame
            Same as decoding_errors, but as a DataFrame and with animal identifiers.

        anova_dfs: DataFrame
            Results from ANOVAs, calculated separately for each age.

        """
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0)
        sessions = ["Training", "Reversal"]
        decoding_errors = dict()
        errors_, sessions_, mice_, ages_, time_bins_, = (
            [],
            [],
            [],
            [],
            [],
        )
        for age, cohort_axs, color in zip(ages, axs, age_colors):
            decoding_errors[age] = {key: [] for key in sessions}
            for mouse in self.meta["grouped_mice"][age]:
                errors = self.plot_reversal_decoding_error(
                    mouse,
                    classifier=classifier,
                    n_spatial_bins=n_spatial_bins,
                    error_time_bin_size=error_time_bin_size,
                    axs=cohort_axs,
                    color=color,
                    predictors=predictors,
                )
                for session in sessions:
                    mouse_means = [np.nanmean(error) for error in errors[session]]
                    n_bins = len(mouse_means)
                    decoding_errors[age][session].append(mouse_means)

                    errors_.extend(mouse_means)
                    sessions_.extend([session for i in range(n_bins)])
                    mice_.extend([mouse for i in range(n_bins)])
                    ages_.extend([age for i in range(n_bins)])
                    time_bins_.extend(np.arange(n_bins))

            for session, session_ax in zip(["Training", "Reversal"], cohort_axs):
                errors_arr = np.vstack(decoding_errors[age][session])
                m = np.nanmean(errors_arr, axis=0)
                se = sem(errors_arr, axis=0)
                session_ax.errorbar(range(len(m)), m, yerr=se, color=color, capsize=1)

        fig.supylabel("Mean decoding error (spatial bins)")
        fig.supxlabel("Time bins")
        df = pd.DataFrame(
            {
                "mice": mice_,
                "age": ages_,
                "session_pairs": sessions_,
                "time_bins": time_bins_,
                "decoding_errors": errors_,
            }
        )

        anova_dfs = {
            age: pg.rm_anova(
                df.loc[df["age"] == age],
                dv="decoding_errors",
                within=["time_bins", "session_pairs"],
                subject="mice",
            )
            for age in ages
        }

        return decoding_errors, df, anova_dfs

    def find_decoding_error(
        self,
        mouse,
        training_and_test_sessions,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        show_plot=True,
        error_time_bin_size=300,
        predictors="cells",
    ):
        """
        Find decoding error between predicted and real spatially
        binned location.

        :parameters
        ---
        See decode_place().
        """
        (
            y_predicted,
            predictor_data,
            outcome_data,
            classifier,
            running,
        ) = self.decode_place(
            mouse,
            training_and_test_sessions,
            classifier=classifier,
            n_spatial_bins=n_spatial_bins,
            show_plot=False,
            predictors=predictors,
        )

        d = get_circular_error(
            y_predicted, outcome_data["test"], n_spatial_bins=n_spatial_bins
        )
        d = d.astype(float)
        d[~running[1]] = np.nan

        bins = make_bins(d, error_time_bin_size, axis=0)
        binned_d = np.split(d, bins)

        if show_plot:
            fig, ax = plt.subplots()
            time_bins = range(len(binned_d))
            mean_errors = [np.nanmean(dist) for dist in binned_d]
            sem_errors = [np.nanstd(dist) / np.sqrt(dist.shape[0]) for dist in binned_d]
            ax.errorbar(time_bins, mean_errors, yerr=sem_errors)

        return binned_d, d

    def all_session_pairs_decoding_error(
        self, mouse, classifier=BernoulliNB(), n_spatial_bins=36
    ):
        """
        For each session pair, compute the mean decoding error and plot it in a matrix.

        """
        shape = (5, 5)
        decoding_error_matrix = nan_array(shape)
        for i, session_pair in enumerate(product(self.meta["session_types"], repeat=2)):
            if session_pair[0] != session_pair[1]:
                decoding_error = self.find_decoding_error(
                    mouse,
                    session_pair,
                    classifier=classifier,
                    n_spatial_bins=n_spatial_bins,
                    show_plot=False,
                )[1]
                row, col = np.unravel_index(i, shape)
                decoding_error_matrix[row, col] = np.nanmean(decoding_error)

        return decoding_error_matrix

    def spatial_decoding_error_matrix(
        self,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        overwrite=False,
        saved_data=r"Z:\Will\RemoteReversal\Data\cross_session_spatial_decoding_error.pkl",
        show_plot=True,
    ):
        if overwrite:
            decoding_error_matrix = dict()
            for age in ages:
                mice_this_age = self.meta["grouped_mice"][age]
                decoding_error_matrix[age] = nan_array((len(mice_this_age), 5, 5))

                for i, mouse in enumerate(self.meta["grouped_mice"][age]):
                    decoding_error_matrix[age][
                        i
                    ] = self.all_session_pairs_decoding_error(
                        mouse, classifier=classifier, n_spatial_bins=n_spatial_bins
                    )

            with open(saved_data, "wb") as file:
                pkl.dump(decoding_error_matrix, file)
        else:
            with open(saved_data, "rb") as file:
                decoding_error_matrix = pkl.load(file)

        errors = dict()
        for age in ["young", "aged"]:
            errors[age] = []
            for lag in np.arange(1, 5):
                errors[age].append(
                    np.diagonal(
                        decoding_error_matrix[age], offset=lag, axis1=1, axis2=2
                    ).flatten()
                )

        if show_plot:
            fig, ax = plt.subplots()
            for age, age_color in zip(ages, age_colors):
                ax.errorbar(
                    np.arange(1, 5),
                    [np.nanmean(x) for x in errors[age]],
                    yerr=[sem(x) for x in errors[age]],
                    color=age_color,
                    capsize=2,
                    linewidth=3,
                )
                for i, data_points in enumerate(errors[age]):
                    ax.scatter(
                        jitter_x(np.ones_like(data_points) * (i + 1.15), 0.03),
                        data_points,
                        color=age_color,
                        edgecolor="k",
                        alpha=0.5,
                    )

            ax.set_xlabel("Days apart")
            ax.set_ylabel("Mean decoding error \n across session pairs [spatial bins]")
            ax.set_xticks(np.arange(1, 5))

        return decoding_error_matrix, errors

    def compare_decoding_accuracy(self, training_test_session):
        decoding_error_matrix, errors = self.spatial_decoding_error_matrix(
            show_plot=False
        )

        row = self.meta["session_types"].index(training_test_session[0])
        col = self.meta["session_types"].index(training_test_session[1])

        session_pair_errors = {}
        for age in ages:
            session_pair_errors[age] = decoding_error_matrix[age][:, row, col]

        fig, ax = plt.subplots(figsize=(3.3, 4.8))
        boxes = ax.boxplot(
            [session_pair_errors[age] for age in ages],
            labels=ages,
            widths=0.75,
            patch_artist=True,
            zorder=0,
        )
        for i, (age, color) in enumerate(zip(ages, age_colors)):
            ax.scatter(
                jitter_x(np.ones_like(session_pair_errors[age]) * (i + 1)),
                session_pair_errors[age],
                edgecolor="k",
                color=color,
                zorder=1,
            )
        color_boxes(boxes, age_colors)
        ax.set_ylabel("Mean decoding error [spatial bins]")
        ax.set_title(
            f"Trained on {training_test_session[0]} "
            f"\n Tested on {training_test_session[1]}"
        )

        return session_pair_errors

    def spatial_decoding_anova(
        self,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        overwrite=False,
        saved_data=r"Z:\Will\RemoteReversal\Data\cross_session_spatial_decoding_error.pkl",
        show_plot=True,
    ):
        decoding_error_matrix, errors_sorted = self.spatial_decoding_error_matrix(
            classifier=classifier,
            n_spatial_bins=n_spatial_bins,
            overwrite=overwrite,
            saved_data=saved_data,
            show_plot=show_plot,
        )

        lags, errors_long = dict(), dict()
        for age in ages:
            lags[age] = []
            errors_long[age] = []
            for lag, errors in enumerate(errors_sorted[age]):
                lags[age].extend(np.ones_like(errors) * (lag + 1))
                errors_long[age].extend(errors)

        ages_long = np.hstack(
            [[age for i in range(len(errors_long[age]))] for age in ages]
        )
        errors_long = np.hstack([errors_long[age] for age in ages])
        lags = np.hstack([lags[age] for age in ages])

        df = pd.DataFrame({"errors": errors_long, "lags": lags, "ages": ages_long})

        anova_df = pg.anova(df, dv="errors", between=["lags", "ages"])
        pairwise_df = df.pairwise_ttests(
            dv="errors", between=["lags", "ages"], padjust="fdr_bh"
        )

        return anova_df, pairwise_df, df

    ############################ ENSEMBLE FUNCTIONS ############################
    def count_ensembles(self, grouped=True, normalize=True,
                        sessions_to_plot=None):
        """
        Plot number of ensembles for each session. Can also normalize by total number of neurons.

        :parameters
        ---
        grouped: bool
            Whether to group into aged vs young mice. If False, plot all mice as lines. If True, make boxplots for
            aged vs young for each session.

        normalize: bool
            Whether to normalize by number of recorded neurons.

        """
        ylabel = (
            "# of ensembles"
            if not normalize
            else "# of ensembles normalized by cell count"
        )

        if sessions_to_plot is None:
            sessions_to_plot = self.meta['session_types']
        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in sessions_to_plot
        ]

        df = pd.DataFrame(index=self.meta['mice'])
        for session_type in sessions_to_plot:
            n_ensembles_sessions = []
            for mouse in self.meta['mice']:
                session = self.data[mouse][session_type]
                n_ensembles_mouse = session.assemblies['significance'].nassemblies
                if normalize:
                    n_ensembles_mouse /= session.imaging['n_neurons']

                n_ensembles_sessions.append(n_ensembles_mouse)

            df[session_type] = n_ensembles_sessions

        df['aged'] = [self.meta['aged'][mouse] for mouse in df.index]
        df = df.sort_values('aged')

        if grouped:
            # Make a dictionary that's [session_type][age] = list of ensemble counts.
            n_ensembles = dict()

            for session_type in sessions_to_plot:
                n_ensembles[session_type] = dict()

                for age in ages:
                    aged = age == 'aged'
                    n_ensembles[session_type][age] = df[session_type].loc[df['aged'] == aged]

            # Plot.
            fig, axs = plt.subplots(1, len(self.meta["session_types"]))
            fig.subplots_adjust(wspace=0)

            for i, (ax, session_type, title) in enumerate(
                zip(axs, sessions_to_plot, session_labels)
            ):
                self.scatter_box(n_ensembles[session_type], ax=ax)
                ax.set_title(title)

                if i > 0:
                    ax.set_yticks([])
                else:
                    ax.set_ylabel(ylabel)

            self.set_age_legend(fig)

        else:
            n_ensembles = nan_array(
                (len(self.meta["mice"]), len(self.meta["session_types"]))
            )
            for i, mouse in enumerate(self.meta["mice"]):
                for j, session_type in enumerate(self.meta["session_types"]):
                    session = self.data[mouse][session_type]
                    n = session.assemblies["significance"].nassemblies

                    if normalize:
                        n /= session.imaging["n_neurons"]

                    n_ensembles[i, j] = n

            fig, ax = plt.subplots()
            for n, mouse in zip(n_ensembles, self.meta["mice"]):
                color = "k" if mouse in self.meta["grouped_mice"]["young"] else "r"
                ax.plot(self.meta["session_labels"], n, color=color)
                ax.annotate(mouse, (0.1, n[0] + 1))

            ax.set_ylabel(ylabel)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            fig.subplots_adjust(bottom=0.2)

        return n_ensembles, df

    def count_unique_ensemble_members(self, session_type, filter_method="sd", thresh=2):
        proportion_unique_members = {age: [] for age in ages}
        ensemble_size = {age: [] for age in ages}
        for age in ages:
            proportion_unique_members[age] = []

            for mouse in self.meta["grouped_mice"][age]:
                patterns = self.data[mouse][session_type].assemblies["patterns"]
                n_neurons = self.data[mouse][session_type].imaging["n_neurons"]

                members = find_members(
                    patterns, filter_method=filter_method, thresh=thresh
                )[1]
                ensemble_size[age].append(
                    np.median(
                        [
                            len(members_this_ensemble)
                            for members_this_ensemble in members
                        ]
                    )
                    / n_neurons
                )
                proportion_unique_members[age].append(
                    len(np.unique(np.hstack(members))) / n_neurons
                )

        return proportion_unique_members, ensemble_size

    def find_promiscuous_neurons(
        self, session_type, filter_method="sd", thresh=2, p_ensemble_thresh=0.1
    ):
        p_promiscuous_neurons = {age: [] for age in ages}
        for age in ages:
            p_promiscuous_neurons[age] = []

            for mouse in self.meta["grouped_mice"][age]:
                patterns = self.data[mouse][session_type].assemblies["patterns"]
                n_ensembles, n_neurons = patterns.shape
                memberships = find_memberships(
                    patterns, filter_method=filter_method, thresh=thresh
                )

                ensemble_threshold = p_ensemble_thresh * n_ensembles

                p_promiscuous_neurons[age].append(
                    sum(
                        np.asarray([len(ensemble) for ensemble in memberships])
                        > ensemble_threshold
                    )
                    / n_neurons
                )

        return p_promiscuous_neurons

    def split_session_ensembles(self, mouse, session_type, overwrite_ensembles=False):
        """
        Split a session in half then look for ensembles separately for each half.

        """
        session = self.data[mouse][session_type]
        if session.meta['local']:
            folder = get_equivalent_local_path(session.meta['folder'])
        else:
            folder = session.meta['folder']
        fpath = os.path.join(folder, 'SplitSessionEnsembles.pkl')

        try:
            if overwrite_ensembles:
                print(f"Overwriting {fpath}")
                raise Exception
            with open(fpath, "rb") as file:
                split_ensembles = pkl.load(file)
        except:
            processed_for_assembly_detection = preprocess_multiple_sessions(
                [session.imaging["S"]], smooth_factor=5, use_bool=True
            )
            split_data = np.array_split(processed_for_assembly_detection["processed"][0], 2, axis=1)

            split_ensembles = {half: find_assemblies(data, nullhyp='circ', plot=False,
                                                     n_shuffles=500)
                               for half, data in zip(['first', 'second'], split_data)}

            with open(fpath, "wb") as file:
                pkl.dump(split_ensembles, file)

        return split_ensembles

    def find_ensemble_port_locations(self, mouse, session_type):
        session = self.data[mouse][session_type]
        n_assemblies = session.assemblies["significance"].nassemblies
        ensemble_field_COMs = [
            session.get_ensemble_field_COM(i) for i in range(n_assemblies)
        ]
        port_locations = np.asarray(session.behavior.data["lin_ports"])

        d = []
        for ensemble in ensemble_field_COMs:
            d.append(get_circular_error(ensemble, port_locations, 2 * np.pi))
        d = np.vstack(d)

        closest_ports = np.argmin(d, axis=1)

        return d, closest_ports

    def ensemble_density_increase(self, mouse, session_pair=("Goals4", "Reversal")):
        # First, find the ports that ensembles fired closest to.
        d = dict()
        closest_ports = dict()
        for session in session_pair:
            d[session], closest_ports[session] = self.find_ensemble_port_locations(
                mouse, session
            )

        # Get the port numbers of the future rewarded ports during
        # Reversal.
        future_rewards = np.where(
            self.data[mouse][session_pair[1]].behavior.data["rewarded_ports"]
        )[0]

        # Find proportion of ensembles that fire at those ports during both
        # sessions.
        p_firing_at_future_reward = np.sum(
            np.in1d(closest_ports[session_pair[0]], future_rewards)
        ) / len(closest_ports[session_pair[0]])

        # Find proportion of ensembles that fire at those ports during Reversal.
        p_firing_at_current_reward = np.sum(
            np.in1d(closest_ports[session_pair[1]], future_rewards)
        ) / len(closest_ports[session_pair[1]])

        return p_firing_at_future_reward, p_firing_at_current_reward

    def ensemble_density_increase_all_mice(self, session_pair=("Goals4", "Reversal")):
        ens_density = dict()
        labels = [session.replace("Goals", "Training") for session in session_pair]
        for age in ages:
            ens_density[age] = nan_array((len(self.meta["grouped_mice"][age]), 2))

            for i, mouse in enumerate(self.meta["grouped_mice"][age]):
                pre, post = self.ensemble_density_increase(mouse, session_pair)
                ens_density[age][i] = np.hstack((pre, post))

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6.5, 7))
        for ax, age, c in zip(axs, ages, age_colors):
            ax.plot(labels, ens_density[age].T, color=c)
            ax.set_xticklabels(labels, rotation=45)
        axs[0].set_ylabel("Proportion of ensembles active near Reversal goals")

    def plot_proportion_fading_ensembles_near_important_ports(
        self, show_plot=True, alpha=0.01, decompose_relevant_ports=False
    ):
        p_near = dict()
        port_types = ["rewarded", "previously_rewarded", "other", "relevant"]

        for age in ages:
            p_near[age] = {key: [] for key in port_types}
            for mouse in self.meta["grouped_mice"][age]:
                # Find the closest ports for each ensemble.
                closest_ports = self.find_ensemble_port_locations(mouse, "Reversal")[1]

                trends = self.find_activity_trends(
                    mouse, "Reversal", data_type="ensembles", alpha=alpha
                )[0]

                port_locations = {
                    "rewarded": self.data[mouse]["Reversal"].behavior.data[
                        "rewarded_ports"
                    ],
                    "previously_rewarded": self.data[mouse]["Goals4"].behavior.data[
                        "rewarded_ports"
                    ],
                }
                port_locations["other"] = ~(
                    port_locations["previously_rewarded"] + port_locations["rewarded"]
                )
                port_locations["relevant"] = ~port_locations["other"]

                port_locations = {
                    key: np.where(value)[0] for key, value in port_locations.items()
                }

                try:
                    for port_type in port_types:
                        p_near[age][port_type].append(
                            np.sum(
                                i in port_locations[port_type]
                                for i in closest_ports[trends["decreasing"]]
                            )
                            / len(trends["decreasing"])
                        )

                except ZeroDivisionError:
                    for port_type in port_types:
                        p_near[age][port_type].append(0)

        if show_plot:
            ports_to_plot = (
                ["rewarded", "previously_rewarded", "other"]
                if decompose_relevant_ports
                else ["relevant", "other"]
            )
            fig, axs = plt.subplots(
                1, len(ports_to_plot), sharey=True, figsize=(7.6, 6.75)
            )
            fig.subplots_adjust(wspace=0)

            if decompose_relevant_ports:
                titles = [
                    "near rewarded ports",
                    "near previously \nrewarded ports",
                    "near other ports",
                ]
            else:
                titles = ["near relevant ports", "near other ports"]

            for ax, port, title in zip(axs, ports_to_plot, titles):
                box = ax.boxplot(
                    [p_near[age][port] for age in ages],
                    labels=ages,
                    widths=0.75,
                    patch_artist=True,
                    zorder=0,
                    showfliers=False,
                )
                for patch, med, color in zip(box["boxes"], box["medians"], age_colors):
                    patch.set_facecolor(color)
                    med.set(color="k")

                for i, (age, color) in enumerate(zip(ages, age_colors)):
                    ax.scatter(
                        jitter_x(np.ones_like(p_near[age][port]) * (i + 1), 0.1),
                        p_near[age][port],
                        color=color,
                        zorder=1,
                        edgecolor="k",
                        s=70,
                    )

                ax.set_title(title)

            fig.suptitle("Fading ensembles")
            axs[0].set_ylabel("Proportion")

        return p_near

    def plot_ensemble_port_locations(self, session_type, show_plot=True):
        near_currently_rewarded = dict()
        near_never_rewarded = dict()
        near_previously_rewarded = dict()

        for age in ages:
            near_currently_rewarded[age] = []
            near_never_rewarded[age] = []
            near_previously_rewarded[age] = []

            for mouse in self.meta["grouped_mice"][age]:
                session = self.data[mouse][session_type]
                d, closest_ports = self.find_ensemble_port_locations(
                    mouse, session_type
                )
                n_ensembles = closest_ports.shape[0]

                if session_type == "Reversal":
                    previous_reward_ports = self.data[mouse]["Goals4"].behavior.data[
                        "rewarded_ports"
                    ]
                else:
                    previous_reward_ports = [False for i in range(8)]
                current_rewarded_ports = session.behavior.data["rewarded_ports"]
                never_rewarded_ports = ~(previous_reward_ports + current_rewarded_ports)

                current_rewarded_ports = np.where(current_rewarded_ports)[0]
                never_rewarded_ports = np.where(never_rewarded_ports)[0]
                previously_rewarded_ports = np.where(previous_reward_ports)[0]

                near_currently_rewarded[age].append(
                    np.sum(
                        [
                            ensemble_port in current_rewarded_ports
                            for ensemble_port in closest_ports
                        ]
                    )
                    / n_ensembles
                )

                near_never_rewarded[age].append(
                    np.sum(
                        [
                            ensemble_port in never_rewarded_ports
                            for ensemble_port in closest_ports
                        ]
                    )
                    / n_ensembles
                )

                near_previously_rewarded[age].append(
                    np.sum(
                        [
                            ensemble_port in previously_rewarded_ports
                            for ensemble_port in closest_ports
                        ]
                    )
                    / n_ensembles
                )

        if show_plot:
            fig, axs = plt.subplots(1, 2, sharey=True)

            for ax, age in zip(axs, ages):
                bottom = np.zeros_like(near_currently_rewarded[age])

                mice = self.meta["grouped_mice"][age]
                for p, label, c in zip(
                    [
                        near_currently_rewarded,
                        near_never_rewarded,
                        near_previously_rewarded,
                    ],
                    ["Currently rewarded", "Never rewarded", "Previously rewarded"],
                    ["g", "r", "y"],
                ):
                    ax.bar(mice, p[age], bottom=bottom, label=label, color=c)
                    bottom += p[age]

                ax.set_xticklabels(mice, rotation=45)

            axs[0].set_ylabel("Proportion")
            axs[1].legend()

    def fit_ensemble_lick_decoder(
        self,
        mouse: str,
        session_types: tuple,
        classifier=RandomForestClassifier,
        licks_to_include="all",
        lag=0,
        fps=15,
        **classifier_kwargs,
    ):
        registered_ensembles = self.match_ensembles(mouse, session_types)
        poor_matches = registered_ensembles["poor_matches"]

        # Gather predictor neural data.
        X = dict()
        y = dict()
        for session_activations, t, session_type in zip(
            registered_ensembles["matched_activations"],
            ["train", "test"],
            session_types,
        ):
            licks = np.asarray(
                self.data[mouse][session_type].behavior.data["df"]["lick_port"]
            )
            activations = zscore(
                np.squeeze(session_activations[~poor_matches]), axis=0
            ).T
            lick_bool = licks > -1
            lick_inds = np.where(lick_bool)[0]
            activation_inds = lick_inds + np.round(lag * fps).astype(int)

            # if licks_to_include=='first':
            #     lick_inds = np.where(np.diff(licks, prepend=0)>0)[0]
            #     activation_inds = lick_inds + np.round(lag*fps).astype(int)

            if licks_to_include == "first":
                licks_only = licks[lick_bool]
                switched_ports = np.diff(licks_only, prepend=0)
                lick_ts = np.where(lick_bool)[0]

                lick_inds = lick_ts[np.where(switched_ports != 0)[0]]
                activation_inds = lick_inds + np.round(lag * fps).astype(int)

                # Handles edge cases where indices exceed the recording duration.
                activation_inds[activation_inds > activations.shape[0]] = (
                    activations.shape[0] - 1
                )
                activation_inds[activation_inds < 0] = 0

            if licks_to_include == "all" or licks_to_include == "first":
                licks = licks[lick_inds]
                activations = activations[activation_inds]
            else:
                pass

            X[t] = activations
            y[t] = licks

        classifier = classifier(**classifier_kwargs)
        classifier.fit(X["train"], y["train"])

        return X, y, classifier

    def registered_ensemble_lick_decoder(
        self,
        mouse,
        session_types,
        classifier=RandomForestClassifier,
        licks_to_include="all",
        n_splits=6,
        lag=0,
        show_plot=True,
        ax=None,
        **classifier_kwargs,
    ):
        X, y, classifier = self.fit_ensemble_lick_decoder(
            mouse,
            session_types,
            classifier=classifier,
            lag=lag,
            licks_to_include=licks_to_include,
            **classifier_kwargs,
        )

        X_split = np.array_split(X["test"], n_splits)
        y_split = np.array_split(y["test"], n_splits)
        scores = [
            classifier.score(X_chunk, y_chunk)
            for X_chunk, y_chunk in zip(X_split, y_split)
        ]

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()

            ax.plot(np.linspace(0, 1, n_splits), scores, "k", alpha=0.2)

        return scores

    def registered_ensemble_lick_decoder_accuracy_by_port(
        self,
        mouse,
        session_types,
        classifier=GaussianNB,
        licks_to_include="first",
        n_splits=6,
        lag=0,
        show_plot=True,
        axs=None,
        **classifier_kwargs,
    ):
        X, y, classifier = self.fit_ensemble_lick_decoder(
            mouse,
            session_types,
            classifier=classifier,
            lag=lag,
            licks_to_include=licks_to_include,
            **classifier_kwargs,
        )

        session = self.data[mouse][session_types[1]]
        reward_locations = np.where(session.behavior.data["rewarded_ports"])[0]
        if session_types[1] == "Reversal":
            previous_session = self.data[mouse]["Goals4"]
            previously_rewarded = np.where(
                previous_session.behavior.data["rewarded_ports"]
            )[0]
        else:
            previously_rewarded = []

        X_split = np.array_split(X["test"], n_splits)
        y_split = np.array_split(y["test"], n_splits)
        port_predictions = [[] for i in range(8)]
        port_accuracies = nan_array((8, n_splits))
        for t, (X_chunk, y_chunk) in enumerate(zip(X_split, y_split)):
            predictions = classifier.predict(X_chunk)

            for port in range(8):
                port_prediction = predictions[y_chunk == port]
                port_predictions[port].append(port_prediction)

                port_accuracies[port, t] = np.sum(port_prediction == port) / len(
                    port_prediction
                )

        if show_plot:
            if axs is None:
                fig, axs = plt.subplots(
                    4, 2, sharey=True, sharex=True, figsize=(6.5, 10)
                )

            for port, (port_accuracy, ax) in enumerate(
                zip(port_accuracies, axs.flatten())
            ):
                if port in reward_locations:
                    title_color = "g"
                elif port in previously_rewarded:
                    title_color = "y"
                else:
                    title_color = "k"
                ax.plot(port_accuracy, alpha=0.5)
                ax.set_title(f"Port #{port}", color=title_color)

            fig.tight_layout()

        return port_accuracies

    def ensemble_lick_anova(
        self,
        classifier=RandomForestClassifier,
        licks_to_include="all",
        n_splits=6,
        session_types=(("Goals3", "Goals4"), ("Goals4", "Reversal")),
        lag=0,
        **classifier_kwargs,
    ):
        """
        Do Bayesian decoding on which reward port is being licked. Then do
        2-way repeated measures ANOVA on young and aged mice.

        :param classifier:
        :param licks_to_include:
        :param n_splits:
        :param session_types:
        :param classifier_kwargs:
        :return:
        """
        scores = dict()
        fig, axs = plt.subplots(
            2, len(session_types), sharey=True, sharex=True, figsize=(12, 8.5)
        )
        time_bins = np.arange(n_splits)

        mice_, decoded_session, ages_, time_bins_ = [], [], [], []
        for age, cohort_ax in zip(ages, axs):
            scores[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                for ax, session_pair in zip(cohort_ax, session_types):
                    scores_ = self.registered_ensemble_lick_decoder(
                        mouse,
                        session_pair,
                        classifier=classifier,
                        licks_to_include=licks_to_include,
                        n_splits=n_splits,
                        show_plot=True,
                        lag=lag,
                        ax=ax,
                        **classifier_kwargs,
                    )

                    scores[age].extend(scores_)
                    decoded_session.extend([session_pair[1] for i in range(n_splits)])
                    mice_.extend([mouse for i in range(n_splits)])
                    ages_.extend([age for i in range(n_splits)])
                    time_bins_.extend(time_bins)

        df = pd.DataFrame(
            {
                "mice": mice_,
                "age": ages_,
                "decoded_session": decoded_session,
                "time_bins": time_bins_,
                "scores": np.hstack([scores[age] for age in ages]),
            }
        )
        means = df.groupby(["age", "decoded_session", "time_bins"]).mean()
        sem = df.groupby(["age", "decoded_session", "time_bins"]).sem()

        for age, row_ax, color in zip(ages, axs, age_colors):
            for session_pair, ax in zip(session_types, row_ax):
                ax.errorbar(
                    np.linspace(0, 1, n_splits),
                    means.loc[age].loc[session_pair[1]].to_numpy(),
                    yerr=np.squeeze(sem.loc[age].loc[session_pair[1]].to_numpy()),
                    capsize=2,
                    color=color,
                )
                ax.set_title(
                    f"{age}, trained on {session_pair[0].replace('Goals', 'Training')}"
                    f"\n tested on {session_pair[1].replace('Goals', 'Training')}"
                )
                ax.axhline(y=1 / 8, color="darkred")
        fig.supxlabel("Time in session (normalized)")
        fig.supylabel("Lick decoding accuracy")
        fig.tight_layout()

        anova_dfs = {
            age: pg.rm_anova(
                df.loc[df["age"] == age],
                dv="scores",
                within=["decoded_session", "time_bins"],
                subject="mice",
            )
            for age in ages
        }

        pairwise_dfs = {
            age: {
                session: pg.pairwise_ttests(
                    dv="scores",
                    within="time_bins",
                    subject="mice",
                    data=df[
                        np.logical_and(
                            df["age"] == age, df["decoded_session"] == session
                        )
                    ],
                    padjust="fdr_bh",
                )
                for session in np.unique(df["decoded_session"].values)
            }
            for age in ages
        }
        return anova_dfs, pairwise_dfs, df

    def ensemble_feature_selector(
        self,
        mouse,
        session_type,
        licks_only=True,
        feature_selector=RFECV,
        estimator=GaussianNB(),
        show_plot=True,
        **kwargs,
    ):
        session = self.data[mouse][session_type]
        ensemble_activations = zscore(session.assemblies["activations"], axis=0).T
        licks = np.asarray(session.behavior.data["df"]["lick_port"])

        if licks_only:
            licking = licks > -1
            ensemble_activations = ensemble_activations[licking]
            licks = licks[licking]

        fs = feature_selector(
            estimator=estimator, cv=StratifiedKFold(3), n_jobs=6, **kwargs
        )
        fs.fit(ensemble_activations, licks)

        # if show_plot:
        #     fig, ax = plt.subplots()
        #     ax.set_xlabel("Number of features selected")
        #     ax.set_ylabel("Cross validation score (nb of correct classifications)")
        #     ax.plot(fs.grid_scores_)

        return fs

    def plot_ensemble_sizes(
        self, filter_method="sd", thresh=2, data_type="unique_members"
    ):
        """
        Plots the relative size of an ensemble (defined by "members") for each session, grouped by age.

        :parameters
        ---
        filter_method: str
            'sd' or 'z' (not done yet).

        thresh: float
            If filter_method=='sd', the number of standard deviations above the mean neuronal weight to be considered
            a member of that ensemble.

        data_type: str
            'unique_members' or 'ensemble_size'
            If 'unique_members', counts all the neurons that are in any ensemble.
            If 'ensemble_size', takes the median across all ensemble sizes (# of members).
        """
        proportion_unique_members = dict()
        ensemble_size = dict()
        for session_type in self.meta["session_types"]:
            (
                proportion_unique_members[session_type],
                ensemble_size[session_type],
            ) = self.count_unique_ensemble_members(
                session_type, filter_method=filter_method, thresh=thresh
            )

        fig, axs = plt.subplots(1, len(self.meta["session_types"]))
        fig.subplots_adjust(wspace=0)

        data = {
            "unique_members": proportion_unique_members,
            "ensemble_size": ensemble_size,
        }
        ylabels = {
            "unique_members": "Unique members / total # neurons",
            "ensemble_size": "Median # members per ensemble / total # neurons",
        }
        to_plot = data[data_type]
        for i, (ax, session_type) in enumerate(zip(axs, self.meta["session_types"])):
            box = ax.boxplot(
                [to_plot[session_type][age] for age in ages],
                labels=ages,
                patch_artist=True,
                widths=0.75,
            )
            for patch, med, color in zip(box["boxes"], box["medians"], age_colors):
                patch.set_facecolor(color)
                med.set(color="k")
            ax.set_xticks([])
            ax.set_title(session_type)

        [ax.set_yticks([]) for ax in axs[1:]]
        axs[0].set_ylabel(ylabels[data_type])

        return data

    def get_lapsed_assembly_activation(
        self,
        mouse,
        session_types,
        smooth_factor=5,
        use_bool=True,
        z_method="global",
        detected="everyday",
    ):
        S_list = self.rearrange_neurons(mouse, session_types, "S", detected=detected)
        data = preprocess_multiple_sessions(
            S_list, smooth_factor=smooth_factor, use_bool=use_bool, z_method=z_method
        )

        lapsed_assemblies = lapsed_activation(data["processed"])

        return lapsed_assemblies

    def plot_lapsed_assemblies(self, mouse, session_types, detected="everyday"):
        lapsed_assemblies = self.get_lapsed_assembly_activation(
            mouse, session_types, detected=detected
        )
        spiking = self.rearrange_neurons(
            mouse, session_types, "spike_times", detected=detected
        )

        n_sessions = len(lapsed_assemblies["activations"])
        for i, pattern in enumerate(lapsed_assemblies["patterns"]):
            fig, axs = plt.subplots(n_sessions, 1)
            for ax, activation, spike_times in zip(
                axs, lapsed_assemblies["activations"], spiking
            ):
                plot_assembly(pattern, activation[i], spike_times, ax=ax)

        return lapsed_assemblies, spiking

    def scrollplot_port_vicinity_activity(
        self,
        mouse,
        session_type,
        inds=None,
        data_type="ensembles",
        time_window=(0.5, 0.5),
        dist_thresh=3,
    ):
        # Abbreviate references.
        session = self.data[mouse][session_type]
        df = session.behavior.data["df"]
        licks = df["lick_port"]
        n_trials = max(df["trials"] + 1)
        trials = np.asarray(df["trials"])
        distance_to_port = distance.cdist(
            np.vstack((df["x"], df["y"])).T, session.behavior.data["ports"]
        )
        rewarded = np.where(session.behavior.data["rewarded_ports"])[0]

        if data_type == "ensembles":
            neural_data = session.assemblies["activations"]
        elif data_type in ["C", "S", "S_binary"]:
            neural_data = session.imaging[data_type]
        else:
            raise ValueError

        if inds is None:
            inds = range(neural_data.shape[0])
        neural_data = neural_data[inds]
        n_frames = neural_data.shape[1]

        # If looking at the Reversal session, also
        if session_type == "Reversal":
            previously_rewarded = np.where(
                self.data[mouse]["Goals4"].behavior.data["rewarded_ports"]
            )[0]
        else:
            previously_rewarded = []

        fps = 15
        t_xaxis = np.arange(-time_window[0], time_window[1] + 1 / fps, 1 / fps)
        time_window = np.round(np.asarray([t * fps for t in time_window])).astype(int)
        window_size = sum(abs(time_window))

        # For each port, collect timestamps before and after licks. If there was no lick,
        # instead collect timestamps around when the mouse arrived at that port.
        all_activity = []
        n_lick_laps = []
        for unit, activity in enumerate(neural_data):
            port_activations = []

            for port in range(8):
                lick_activations = []
                passed_activations = []

                for trial in range(n_trials):
                    # Get timestamps when the mouse licked if at all.
                    on_trial = trials == trial
                    licks_ = np.where(np.logical_and(licks == port, on_trial))[0]
                    did_not_lick = licks_.size == 0

                    # If the mouse didn't lick, get the first timestamp when the mouse got close
                    # to the port.
                    if did_not_lick:
                        near_port = np.where(
                            np.logical_and(
                                on_trial, distance_to_port[:, port] < dist_thresh
                            )
                        )[0]

                        if len(near_port) > 0:
                            t0 = near_port[0]
                        else:
                            activation = nan_array(window_size)
                            passed_activations.append(activation)
                            continue

                    else:
                        t0 = licks_[0]

                    pre = t0 - time_window[0]
                    post = t0 + time_window[1]
                    # Handles edge cases where the window extends past the session start
                    # or end. In those cases, pad with nans.
                    front_pad, back_pad = 0, 0
                    if pre < 0:
                        front_pad = 0 - pre
                        pre = 0
                    if post > n_frames:
                        back_pad = post - n_frames
                        post = n_frames
                    window_ind = slice(pre, post)
                    activation = np.pad(
                        activity[window_ind],
                        (front_pad, back_pad),
                        mode="constant",
                        constant_values=np.nan,
                    )

                    if did_not_lick:
                        passed_activations.append(activation)
                    else:
                        lick_activations.append(activation)

                # Handles edge cases where the mouse didn't lick at a particular port.
                if not lick_activations:
                    port_activation = passed_activations
                else:
                    port_activation = np.vstack((lick_activations, passed_activations))

                if len(n_lick_laps) < 8:
                    n_lick_laps.append(len(lick_activations))
                port_activations.append(port_activation)

            all_activity.append(np.asarray(port_activations))

        titles = [f"Ensemble #{i}" for i in inds]
        if data_type in ["C", "S", "S_binary"]:
            titles = [title.replace("Ensemble", "Neuron") for title in titles]
        ScrollObj = ScrollPlot(
            plot_port_activations,
            current_position=0,
            nrows=4,
            ncols=2,
            subplot_kw={
                "projection": "rectilinear",
                "sharey": True,
                "sharex": True,
            },
            figsize=(7, 10.5),
            port_activations=all_activity,
            t_xaxis=t_xaxis,
            rewarded=rewarded,
            previously_rewarded=previously_rewarded,
            n_lick_laps=n_lick_laps,
            titles=titles,
        )

        return all_activity

    def scrollplot_registered_neurons_port_vicinity(
        self,
        mouse,
        reference_session,
        plot_session,
        neurons_from_reference=None,
        detected="everyday",
    ):
        trimmed_map = self.get_cellreg_mappings(
            mouse,
            (reference_session, plot_session),
            detected=detected,
            neurons_from_session1=neurons_from_reference,
        )[0]

        self.scrollplot_port_vicinity_activity(
            mouse, plot_session, inds=trimmed_map.iloc[:, 1].to_numpy(), data_type="S"
        )

    def find_activity_trends(
        self,
        mouse,
        session_type,
        x="trial",
        z_threshold=None,
        x_bin_size=6,
        data_type="ensembles",
        subset=None,
        alpha=0.01,
    ):
        """
        Find assembly "trends", whether they are increasing/decreasing in occurrence rate over the course
        of a session. Use the Mann Kendall test to find trends.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_type: str
            One of self.meta['session_types'].

        x: str, 'time' or 'trial'
            Whether you want to bin assembly activations by trial # or raw time.

        z_threshold: float
            Z-score threshold at which we consider an assembly to be active.

        x_bin_size: int
            If x == 'time', bin size in seconds.
            If x == 'trial', bin size in trials.

        :return
        ---
        assembly_trends: dict
            With keys 'increasing', 'decreasing' or 'no trend' containing lists of assembly indices.
        """
        session = self.data[mouse][session_type]

        if data_type == "ensembles":
            data = zscore(session.assemblies["activations"], nan_policy="omit", axis=1)
        elif data_type in ["S", "C"]:
            data = zscore(session.imaging[data_type], nan_policy="omit", axis=1)
        elif data_type == "S_binary":
            data = session.imaging[data_type]
        else:
            raise ValueError
        slopes = nan_array(data.shape[0])
        if subset is not None:
            data = data[subset]

        # Binarize activations so that there are no strings of 1s spanning multiple frames.
        activations = np.zeros_like(data)
        if z_threshold is not None:
            for i, unit in enumerate(data):
                on_frames = contiguous_regions(unit > z_threshold)[:, 0]
                activations[i, on_frames] = 1
        else:
            activations = data

        # If binning by time, sum activations every few seconds or minutes.
        if x == "time":
            binned_activations = []
            for unit in activations:
                binned_activations.append(
                    bin_transients(
                        unit[np.newaxis],
                        x_bin_size,
                        fps=15,
                        non_binary=True if z_threshold is None else False,
                    )[0]
                )

            binned_activations = np.vstack(binned_activations)

        # If binning by trials, sum activations every n trials.
        elif x == "trial":
            trial_bins = np.arange(0, session.behavior.data["ntrials"], x_bin_size)
            df = session.behavior.data["df"]

            binned_activations = nan_array((activations.shape[0], len(trial_bins) - 1))
            for i, (lower, upper) in enumerate(zip(trial_bins, trial_bins[1:])):
                in_trial = (df["trials"] >= lower) & (df["trials"] < upper)
                binned_activations[:, i] = np.nanmean(activations[:, in_trial], axis=1)

        else:
            raise ValueError("Invalid value for x.")

        # Group cells/ensembles into either increasing, decreasing, or no trend in occurrence rate.
        trends = {
            "no trend": [],
            "decreasing": [],
            "increasing": [],
        }
        if subset is not None:
            for i, unit in zip(subset, binned_activations):
                mk_test = mk.original_test(unit, alpha=alpha)
                trends[mk_test.trend].append(i)
                slopes[i] = mk_test.slope
        else:
            for i, unit in enumerate(binned_activations):
                mk_test = mk.original_test(unit, alpha=alpha)
                trends[mk_test.trend].append(i)
                slopes[i] = mk_test.slope

        return trends, binned_activations, slopes

    def compare_trend_slopes(
        self, mouse, session_pair=("Goals4", "Reversal"), **kwargs
    ):
        trends = dict()
        slopes = dict()

        registered_ensembles = self.match_ensembles(mouse, session_pair)
        for session_type in session_pair:
            trends[session_type], _, slopes[session_type] = self.find_activity_trends(
                mouse, session_type, **kwargs
            )

        fading_ensembles = np.asarray(trends[session_pair[0]]["decreasing"])
        if fading_ensembles.size == 0:
            return [], []
        reference_ensembles = fading_ensembles[
            ~registered_ensembles["poor_matches"][fading_ensembles]
        ]
        fading_ensembles_yesterday = registered_ensembles["matches"][
            reference_ensembles
        ]
        x = slopes[session_pair[0]][fading_ensembles]
        y = slopes[session_pair[1]][fading_ensembles_yesterday]

        return x, y

    def compare_all_fading_ensemble_slopes(self, **kwargs):
        reversal_slopes = []
        training4_slopes = []
        fig, ax = plt.subplots()
        for age, color in zip(ages, age_colors):
            for mouse in self.meta["grouped_mice"][age]:
                x, y = self.compare_trend_slopes(mouse, **kwargs)
                ax.scatter(x, y, edgecolor=color)

                ax.set_xlabel("Ensemble activation slope on Reversal")
                ax.set_ylabel("Ensemble activation slope on Training4")

                reversal_slopes.extend(x)
                training4_slopes.extend(y)

        ax.axis("equal")

        return reversal_slopes, training4_slopes

    def find_proportion_changing_cells(
        self,
        mouse,
        session_type,
        ensemble_trends=None,
        ensemble_trend="decreasing",
        cell_trend="decreasing",
        **kwargs,
    ):
        """
        Calculate the proportion of cells that follow a particular trend (no trend, decreasing, or increasing) that are members in
        ensembles that follow another trend (no trend, decreasing, or increasing).

        """
        # Give the option to pre-compute this.
        if ensemble_trends is None:
            ensemble_trends = self.find_activity_trends(mouse, session_type, **kwargs)[
                0
            ]

        patterns = self.data[mouse][session_type].assemblies["patterns"]
        prop_changing_cells = []
        for fading_ensemble in ensemble_trends[ensemble_trend]:
            ensemble_members = find_members(patterns[fading_ensemble])[1]
            cell_trends = self.find_activity_trends(
                mouse, session_type, **kwargs, data_type="S", subset=ensemble_members
            )[0]

            prop_changing_cells.append(
                len(cell_trends[cell_trend]) / len(ensemble_members)
            )

        return prop_changing_cells

    def plot_proportion_fading_cells_in_ensembles(
        self,
        session_type="Reversal",
        x="trial",
        x_bin_size=6,
        z_threshold=None,
    ):

        prop_fading_cells = dict()
        for age in ages:
            prop_fading_cells[age] = {"trendless ensembles": [], "fading ensembles": []}
            for mouse in self.meta["grouped_mice"][age]:
                ensemble_trends = self.find_activity_trends(
                    mouse,
                    session_type,
                    x=x,
                    x_bin_size=x_bin_size,
                    z_threshold=z_threshold,
                )[0]

                for key, trend in zip(
                    prop_fading_cells[age].keys(), ["no trend", "decreasing"]
                ):
                    prop_fading_cells[age][key].append(
                        self.find_proportion_changing_cells(
                            mouse,
                            session_type,
                            ensemble_trends=ensemble_trends,
                            x=x,
                            x_bin_size=x_bin_size,
                            z_threshold=z_threshold,
                            ensemble_trend=trend,
                        )
                    )
        n_mice = len(self.meta["mice"])
        colors = distinct_colors(n_mice)
        i = 0
        mean0 = lambda x: np.nanmean(x) if x else 0
        fig, axs = plt.subplots(1, 2, sharey=True)
        for age, ax in zip(ages, axs):
            for trendless, fading in zip(
                prop_fading_cells[age]["trendless ensembles"],
                prop_fading_cells[age]["fading ensembles"],
            ):
                x = np.hstack((np.ones_like(trendless), np.ones_like(fading) * 2))
                ax.scatter(
                    jitter_x(x, 0.05),
                    np.hstack((trendless, fading)),
                    alpha=0.2,
                    color=colors[i],
                )
                ax.plot(
                    jitter_x([1, 2]),
                    np.hstack((mean0(trendless), mean0(fading))),
                    "o-",
                    color=colors[i],
                )
                ax.set_xticks([1, 2])
                ax.set_xticklabels(
                    ["Trendless \nensembles", "Fading \nensembles"], rotation=45
                )
                i += 1
                ax.set_title(age)

        axs[0].set_ylabel("Proportion fading cells")
        fig.tight_layout()

        return prop_fading_cells

    def plot_assembly_trends(
        self,
        x="trial",
        x_bin_size=6,
        z_threshold=None,
        data_type="ensembles",
        show_plot=True,
        alpha=0.01,
    ):
        session_types = self.meta["session_types"]
        session_labels = self.meta["session_labels"]
        assembly_trend_arr = np.zeros(
            (
                3,
                len(self.meta["mice"]),
                len(session_types),
            ),
            dtype=object,
        )
        assembly_counts_arr = np.zeros_like(assembly_trend_arr)

        trend_strs = ["no trend", "decreasing", "increasing"]
        for i, mouse in enumerate(self.meta["mice"]):
            for j, session_type in enumerate(session_types):
                assembly_trends = self.find_activity_trends(
                    mouse,
                    session_type,
                    x=x,
                    x_bin_size=x_bin_size,
                    z_threshold=z_threshold,
                    data_type=data_type,
                    alpha=alpha,
                )[0]

                for h, trend in enumerate(trend_strs):
                    assembly_trend_arr[h, i, j] = assembly_trends[trend]
                    assembly_counts_arr[h, i, j] = len(assembly_trends[trend])

        # To extract these values, use tolist()
        assembly_trends = xr.DataArray(
            assembly_trend_arr,
            dims=("trend", "mouse", "session"),
            coords={
                "trend": trend_strs,
                "mouse": self.meta["mice"],
                "session": session_types,
            },
        )

        assembly_counts = xr.DataArray(
            assembly_counts_arr,
            dims=("trend", "mouse", "session"),
            coords={
                "trend": trend_strs,
                "mouse": self.meta["mice"],
                "session": session_types,
            },
        )

        if show_plot:
            fig, ax = plt.subplots()
            p_decreasing = assembly_counts.sel(
                trend="decreasing"
            ) / assembly_counts.sum(dim="trend")

            for c, age in zip(["r", "k"], ages):
                ax.plot(
                    session_labels,
                    p_decreasing.sel(mouse=self.meta["grouped_mice"][age]).T,
                    color=c,
                )
            ax.set_ylabel("Proportion ensembles with decreasing activity over session")
            plt.setp(ax.get_xticklabels(), rotation=45)

        return assembly_trends, assembly_counts

    def plot_proportion_changing_ensembles(
        self,
        x="trial",
        x_bin_size=6,
        z_threshold=None,
        sessions=("Goals4", "Reversal"),
        trend="decreasing",
        data_type="ensembles",
        alpha=0.01,
        show_plot=True,
    ):
        ensemble_trends, ensemble_counts = self.plot_assembly_trends(
            x=x,
            x_bin_size=x_bin_size,
            z_threshold=z_threshold,
            data_type=data_type,
            show_plot=False,
            alpha=alpha,
        )
        p_changing = ensemble_counts.sel(trend=trend) / ensemble_counts.sum(dim="trend")
        p_changing_split_by_age = dict()
        for age in ages:
            p_changing_split_by_age[age] = [
                p_changing.sel(session=session, mouse=self.meta["grouped_mice"][age])
                for session in sessions
            ]

        if show_plot:
            ylabel = {
                "decreasing": "fading",
                "increasing": "rising",
                "no trend": "flat",
            }
            fig, axs = plt.subplots(1, 2, figsize=(7, 6), sharey=True)
            fig.subplots_adjust(wspace=0)
            for ax, age, color in zip(axs, ages, age_colors):
                boxes = ax.boxplot(
                    p_changing_split_by_age[age],
                    patch_artist=True,
                    widths=0.75,
                    zorder=0,
                    showfliers=False,
                )
                for patch, med in zip(boxes["boxes"], boxes["medians"]):
                    patch.set_facecolor(color)
                    med.set(color="k")

                data_points = np.vstack(p_changing_split_by_age[age]).T
                for mouse_data in data_points:
                    ax.plot(
                        jitter_x([1, 2], 0.05),
                        mouse_data,
                        "o-",
                        color="k",
                        markerfacecolor=color,
                        zorder=1,
                        markersize=10,
                    )

                if age == "aged":
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel("Proportion " + ylabel[trend] + " ensembles")
                ax.set_title(age)
                ax.set_xticklabels(
                    [
                        session_type.replace("Goals", "Training")
                        for session_type in sessions
                    ],
                    rotation=45,
                )

        return p_changing_split_by_age

    def make_fading_ensemble_df(self, p_changing_split_by_age):
        p = p_changing_split_by_age
        df = pd.concat(
            [pd.DataFrame(np.asarray(p[age]).T,
                          index=self.meta['grouped_mice'][age],
                          columns=[str(x.session.values) for x in p[age]])
             for age in ages])

        df['aged'] = [self.meta['aged'][mouse] for mouse in df.index]

        return df

    def correlate_prop_changing_ensembles_to_behavior(
        self,
        x="trial",
        x_bin_size=6,
        z_threshold=None,
        trend="decreasing",
        performance_metric="CRs",
        data_type="ensembles",
        alpha=0.01,
        ax=None,
        ages_to_plot=None,
    ):
        ensemble_trends, ensemble_counts = self.plot_assembly_trends(
            x=x,
            x_bin_size=x_bin_size,
            z_threshold=z_threshold,
            data_type=data_type,
            show_plot=False,
            alpha=alpha,
        )
        p_changing = ensemble_counts.sel(trend=trend) / ensemble_counts.sum(dim="trend")

        p_changing_split_by_age = dict()
        for age in ages:
            p_changing_split_by_age[age] = p_changing.sel(
                session="Reversal", mouse=self.meta["grouped_mice"][age]
            )

        performance = self.plot_performance_session_type(
            "Reversal",
            window=None,
            performance_metric=performance_metric,
            show_plot=False,
        )

        if ax is None:
            fig, ax = plt.subplots()

        ylabels = {"CRs": "Correct rejection rate", "hits": "Hit rate", "d_prime": "d'"}
        if ages_to_plot is None:
            ages_to_plot = ages
            plot_colors = age_colors
        else:
            plot_colors = [age_colors[ages.index(ages_to_plot)]]
            if type(ages_to_plot) is str:
                ages_to_plot = [ages_to_plot]

        for age, c in zip(ages_to_plot, plot_colors):
            prop_fading = np.asarray(p_changing_split_by_age[age], dtype=float)
            perf = performance[age]
            ax.scatter(
                prop_fading,
                perf,
                facecolors=c,
                edgecolors="k",
                s=50,
            )
            z = np.polyfit(prop_fading, perf, 1)
            y_hat = np.poly1d(z)(prop_fading)
            ax.plot(prop_fading, y_hat, color=c)

        ax.set_xlabel("Proportion of fading ensembles")
        ax.set_ylabel(ylabels[performance_metric])

        if ages_to_plot is None:
            ax.legend(ages_to_plot, loc="lower right")

        p_changing = np.hstack([p_changing_split_by_age[age] for age in ages])
        perf = np.hstack([performance[age] for age in ages])

        r, pvalue = spearmanr(p_changing, perf)
        msg = f"r = {np.round(r, 3)}, p = {np.round(pvalue, 3)}"
        if pvalue < 0.05:
            msg += "*"
        print(msg)

        return p_changing_split_by_age, performance

    def make_corr_df(self, p_changing_split_by_age, performance):
        p = p_changing_split_by_age
        df = pd.concat(
            [pd.DataFrame(np.asarray(p[age]).T,
                          index=self.meta['grouped_mice'][age],
                          columns=['proportion fading ensembles'])
             for age in ages])

        df['performance'] = np.hstack([performance[age] for age in ages])
        df['aged'] = [self.meta['aged'][mouse] for mouse in df.index]

        return df

    def plot_assembly_by_trend(
        self,
        mouse,
        session_type,
        trend,
        x="trial",
        x_bin_size=6,
        plot_type="temporal",
        thresh=2.5,
        trend_z_threshold=None,
        alpha=0.01,
    ):
        assembly_trends = self.find_activity_trends(
            mouse,
            session_type,
            x=x,
            x_bin_size=x_bin_size,
            z_threshold=trend_z_threshold,
            alpha=alpha,
        )[0]

        session = self.data[mouse][session_type]
        for assembly_number in assembly_trends[trend]:
            if plot_type == "temporal":
                session.plot_assembly(assembly_number)
            elif plot_type == "spiral":
                ax = session.spiralplot_assembly(assembly_number, threshold=thresh)

                if session_type == "Reversal":
                    goals4 = self.data[mouse]["Goals4"].behavior.data
                    reversal = self.data[mouse]["Reversal"].behavior.data
                    for port in np.asarray(reversal["lin_ports"])[
                        goals4["rewarded_ports"]
                    ]:
                        ax.axvline(x=port, color="y")

    def match_ensembles(self, mouse, session_types):
        """
        Match assemblies across two sessions. For each assembly in the first session of the session_types tuple,
        find the corresponding assembly in the second session by taking the highest cosine similarity between two
        assembly patterns.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_types: tuple
            Two session names (e.g. (Goals1, Goals2)) OR one session name twice ('Reversal','Reversal') in which case
            split_session must be True. Order matters.

        absolute_value: boolean
            Whether to take the absolute value of the pattern similarity matrix. Otherwise, try negating the pattern
            and take the larger value of the two resulting cosine similarities.

        :returns
        ---
        registered_ensembles: dict
            {'similarities': (n_ensembles_s1, n_ensembles_s2) array of cosine similarities.
             'matches': (n_ensembles_s1,) array of indices in s2 that s1 ensembles map onto.
             'best_similarities': (n_ensembles_s1,) array of highest cosine similarities for each ensemble in s1.
             'patterns': list of (n_ensembles, n_neurons) weights for each session.
             'matched_patterns': (2, n_ensembles_s1, n_neurons) array of pateerns. Each index in axis 1 corresponds to
                the matched patterns across the two sessions.
             'matched_activations': list of (n_ensembles, t) arrays of activations for each session.
             'z_similarities': (2, n_ensembles_s1, n_ensembles_s2) array of z-scored similarities across a row.
             'poor_matches': (n_ensembles_s1,) array of booleans indicating whether an ensemble was a poor match,
                defined as an ensemble not having a single z-scored cosine similarity above 2.58. 

        """
        split_session = True if len(np.unique(session_types)) == 1 else False
        if split_session:
            split_ensembles = self.split_session_ensembles(mouse, session_types[0])
            rearranged_patterns = [split_ensembles[half]['patterns'] for half in ['first', 'second']]
            patterns_iterable = rearranged_patterns.copy()

            activations = [
                split_ensembles[half]["activations"] for half in ['first', 'second']
            ]
        else:
            # Get the patterns from each session, matching the neurons.
            rearranged_patterns = self.rearrange_neurons(
                mouse, session_types, data_type="patterns", detected="everyday"
            )

            # To keep consistent with its other outputs, self.rearrange_neurons()
            # gives a neuron x something (in this use case, assemblies) matrix.
            # We actually want to iterate over assemblies here, so transpose.
            patterns_iterable = [
                session_patterns.T for session_patterns in rearranged_patterns
            ]
            activations = [
                self.data[mouse][session].assemblies["activations"]
                for session in session_types
            ]

        # For each assembly in session 1, compute its cosine similarity
        # to every other assembly in session 2. Place this in a matrix.
        assembly_numbers = [range(pattern.shape[0]) for pattern in patterns_iterable]
        similarity_matrix_shape = [pattern.shape[0] for pattern in patterns_iterable]

        similarities = np.zeros((2, *similarity_matrix_shape))
        for combination, pattern_pair in zip(
            product(*assembly_numbers), product(*patterns_iterable)
        ):

            # Compute cosine similarity.
            similarities[0, combination[0], combination[1]] = cosine_similarity(
                *[pattern.reshape(1, -1) for pattern in pattern_pair]
            )[0, 0]

        # The cosine similarity of the negated pattern is simply the negated cosine similarity of
        # the non-negated pattern.
        similarities[1] = -similarities[0]

        # Now, for each assembly, find its best match (i.e., argmax the cosine similarity).
        n_assemblies_first_session = rearranged_patterns[0].shape[1]
        assembly_matches = np.zeros(
            n_assemblies_first_session, dtype=int
        )  # (assemblies,) - index of session 2 match
        best_similarities = nan_array(
            assembly_matches.shape
        )  # (assemblies,) - best similarity for each assembly
        z_similarities = np.stack(
            [zscore(similarity) for similarity in similarities], axis=0
        )  # (assemblies, assemblies) - z-scored similarities

        matched_patterns = np.zeros(
            (2, *patterns_iterable[0].shape)
        )  # (2, assemblies, neurons)
        matched_patterns[0] = patterns_iterable[0]
        matched_activations = [
            activations[0],
            np.zeros((n_assemblies_first_session, activations[1].shape[1])),
        ]  # (2, assemblies, time)

        # For each assembly comparison, find the highest cosine similarity and the corresponding assembly index.
        for assembly_number, possible_matches in enumerate(similarities[0]):
            best_similarities[assembly_number] = np.max(possible_matches)
            match = np.argmax(possible_matches)
            assembly_matches[assembly_number] = match
            matched_patterns[1, assembly_number] = patterns_iterable[1][match]
            matched_activations[1][assembly_number] = activations[1][match]

        # If we also negated the patterns, check if any of those were better.
        for assembly_number, possible_matches in enumerate(similarities[1]):
            best_similarity_negated = np.max(possible_matches)

            # If so, replace the similarity value with the higher one, replace the assembly index, and negate the
            # pattern.
            if best_similarity_negated > best_similarities[assembly_number]:
                best_similarities[assembly_number] = best_similarity_negated
                match = np.argmax(possible_matches)
                assembly_matches[assembly_number] = match
                matched_patterns[1, assembly_number] = -patterns_iterable[1][match]
                matched_activations[1][assembly_number] = activations[1][match]

        # If any assembly doesn't have a single match whose z-scored similarity is above 2.58 (p<0.01), it's not a true
        # match. Exclude it.
        registered_ensembles = {
            "similarities": similarities,
            "matches": assembly_matches,
            "best_similarities": best_similarities,
            "patterns": patterns_iterable,
            "matched_patterns": matched_patterns,
            "matched_activations": matched_activations,
            "z_similarities": z_similarities,
            "poor_matches": np.sum(
                [np.any(z > 2.58, axis=1) for z in z_similarities], axis=0
            )
            == 0,
        }

        return registered_ensembles

    def plot_matched_ensemble(
        self,
        registered_activations,
        registered_patterns,
        registered_spike_times,
        similarity,
        session_types,
        poor_match=False,
        split_session=False,
        frames=None,
    ):
        order = np.argsort(np.abs(registered_patterns[0]))

        fig = plt.figure(figsize=(19.2, 10.7))
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        assembly_axs = [fig.add_subplot(spec[i, 0]) for i in range(2)]

        if split_session:
            assembly_axs[0].get_shared_x_axes().join(*assembly_axs)
            assembly_axs[0].get_shared_y_axes().join(*assembly_axs)

        pattern_ax = fig.add_subplot(spec[:, 1])

        if frames is None:
            frames = [range(len(activation)) for activation in registered_activations]

        for ax, activation, spike_times, pattern, c, session, frames_ in zip(
            assembly_axs,
            registered_activations,
            registered_spike_times,
            registered_patterns,
            ["k", "r"],
            session_types,
            frames,
        ):
            # Plot assembly activation.
            plot_assembly(
                0,
                activation,
                spike_times,
                sort_by_contribution=False,
                order=order,
                ax=ax,
                frames=frames_,
            )
            ax.set_title(session)

            # Plot the patterns.
            pattern_ax = plot_pattern(pattern, ax=pattern_ax, color=c, alpha=0.5)

        # Label the patterns.
        pattern_ax.legend(session_types)
        title = f"Cosine similarity: {np.round(similarity, 3)}"
        if poor_match:
            title += " NON-SIG MATCH!"
        pattern_ax.set_title(title)
        pattern_ax.set_xticks([0, registered_patterns[0].shape[0]])
        pattern_ax = beautify_ax(pattern_ax)
        fig.tight_layout()

        return fig

    def plot_matched_ensembles(self, mouse, session_types: tuple, subset="all"):
        split_session = True if len(np.unique(session_types)) == 1 else False
        registered_ensembles = self.match_ensembles(mouse, session_types)

        if split_session:
            session = self.data[mouse][session_types[0]]
            spike_times = session.imaging['spike_times']
            n_frames = len(session.imaging['frames'])

            halfway = np.round(n_frames / 2)

            # Split the spike times down the midpoint of the session.
            registered_spike_times = [[spikes[spikes < halfway] for spikes in spike_times]]
            registered_spike_times.append([spikes[spikes >= halfway] for spikes in spike_times])

            frames = [np.arange(halfway), np.arange(halfway, n_frames)]
        else:
            registered_spike_times = self.rearrange_neurons(
                mouse, session_types, "spike_times", detected="everyday"
            )
            frames = None

        if subset == "all":
            subset = range(registered_ensembles["matched_patterns"].shape[1])

        for (
            s1_activation,
            s2_activation,
            s1_pattern,
            s2_pattern,
            poor_match,
            similarity,
        ) in zip(
            registered_ensembles["matched_activations"][0][subset],
            registered_ensembles["matched_activations"][1][subset],
            registered_ensembles["matched_patterns"][0][subset],
            registered_ensembles["matched_patterns"][1][subset],
            registered_ensembles["poor_matches"][subset],
            registered_ensembles["best_similarities"][subset],
        ):
            self.plot_matched_ensemble(
                (s1_activation, s2_activation),
                (s1_pattern, s2_pattern),
                registered_spike_times,
                similarity,
                session_types,
                poor_match=poor_match,
                split_session=split_session,
                frames=frames,
            )

    def spiralplot_matched_ensembles(
        self, mouse, session_types: tuple, thresh=1, subset="all"
    ):
        # Match assemblies.
        registered_ensembles = self.match_ensembles(mouse, session_types)

        if subset == "all":
            subset = range((len(registered_ensembles["matches"])))

        # Get timestamps and linearized position.
        t = [
            self.data[mouse][session].behavior.data["df"]["t"]
            for session in session_types
        ]
        linearized_position = [
            self.data[mouse][session].behavior.data["df"]["lin_position"]
            for session in session_types
        ]

        # For each assembly in session 1 and its corresponding match in session 2, get their activation profiles.
        for s1_assembly, (s2_assembly, poor_match) in enumerate(
            zip(registered_ensembles["matches"], registered_ensembles["poor_matches"])
        ):
            if s1_assembly in subset:
                activations = [
                    self.data[mouse][session_type].assemblies["activations"][assembly]
                    for session_type, assembly in zip(
                        session_types, [s1_assembly, s2_assembly]
                    )
                ]

                # Make a figure with 2 subplots, one for each session.
                fig, axs = plt.subplots(
                    2, 1, subplot_kw=dict(polar=True), figsize=(6.4, 10)
                )
                for ax, activation, t_, lin_pos, assembly_number, session_type in zip(
                    axs,
                    activations,
                    t,
                    linearized_position,
                    [s1_assembly, s2_assembly],
                    session_types,
                ):

                    # Find activation threshold and plot location of assembly activation.
                    z_activations = zscore(activation)
                    above_thresh = z_activations > thresh
                    ax = spiral_plot(
                        t_,
                        lin_pos,
                        above_thresh,
                        ax=ax,
                        marker_legend="Ensemble activation",
                    )
                    title_color = "r" if poor_match else "k"
                    ax.set_title(
                        f"Ensemble #{assembly_number} in session {session_type}",
                        color=title_color,
                    )

                    behavior_data = self.data[mouse][session_type].behavior.data
                    port_locations = np.asarray(behavior_data["lin_ports"])[
                        behavior_data["rewarded_ports"]
                    ]
                    [ax.axvline(port, color="g") for port in port_locations]

                    # if session_type == "Reversal":
                    #     behavior_data = self.data[mouse]["Goals4"].behavior.data
                    #     # NEED TO FINISH PLOTTING REWARD SITES

        return registered_ensembles

    def plot_pattern_cosine_similarities(self, session_pair: tuple, show_plot=True):
        """
        For a given session pair, plot the distributions of cosine similarities of assemblies for each mouse,
        separated into young versus aged.

        :param session_pair:
        :param show_plot:
        :return:
        """
        similarities = dict()
        for age in ["young", "aged"]:
            similarities[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                registered_ensembles = self.match_ensembles(mouse, session_pair)
                similarities[age].append(
                    registered_ensembles["best_similarities"][
                        ~registered_ensembles["poor_matches"]
                    ]
                )

        if show_plot:
            fig, axs = plt.subplots(1, 2, figsize=(7, 6))
            fig.subplots_adjust(wspace=0)
            for age, color, ax in zip(ages, age_colors, axs):
                mice = self.meta["grouped_mice"][age]
                boxes = ax.boxplot(similarities[age], patch_artist=True)
                for patch, med in zip(boxes["boxes"], boxes["medians"]):
                    patch.set_facecolor(color)
                    med.set(color="k")

                if age == "aged":
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("Cosine similarity of matched assemblies")
                ax.set_xticklabels(mice, rotation=45)

        return similarities

    def percent_matched_ensembles(
        self, session_pair1=("Goals3", "Goals4"), session_pair2=("Goals4", "Reversal")
    ):
        percent_matches = dict()
        for age in ["young", "aged"]:
            percent_matches[age] = dict()

            for session_pair in [session_pair1, session_pair2]:
                percent_matches[age][session_pair] = []
                for mouse in self.meta["grouped_mice"][age]:
                    poor_matches = self.match_ensembles(mouse, session_pair)[
                        "poor_matches"
                    ]
                    percent_matches[age][session_pair].append(
                        np.sum(~poor_matches) / len(poor_matches)
                    )

        fig, axs = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0)

        for age, ax, color in zip(["young", "aged"], axs, age_colors):
            boxes = ax.boxplot(percent_matches[age].values(), patch_artist=True)
            for patch, med in zip(boxes["boxes"], boxes["medians"]):
                patch.set_facecolor(color)
                med.set(color="k")

            if age == "aged":
                ax.set_yticks([])
            else:
                ax.set_ylabel("Percent matched ensembles")
            ax.set_xticklabels([session_pair1, session_pair2], rotation=45)
            ax.set_title(age)
        fig.tight_layout()

        return percent_matches

    def plot_pattern_cosine_similarity_comparisons(
        self, session_pair1=("Goals3", "Goals4"), session_pair2=("Goals4", "Reversal")
    ):
        """
        For two given session pairs, plot the distribution of cosine similarities for each mouse and the two session
        pairs side by side. This is good for assessing pattern similarity across Goals3 vs Goals4 and Goals4 vs Reversal.

        :param session_pair1:
        :param session_pair2:
        :return:
        """
        similarities = dict()
        for session_pair in [session_pair1, session_pair2]:
            similarities[session_pair] = self.plot_pattern_cosine_similarities(
                session_pair, show_plot=False
            )

        fig, axs = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0)

        for age, ax, color in zip(ages, axs, age_colors):
            mice = self.meta["grouped_mice"][age]
            positions = [
                np.arange(start, start + 3 * len(mice), 3) for start in [-0.5, 0.5]
            ]
            label_positions = np.arange(0, 3 * len(mice), 3)

            for session_pair, position in zip(
                [session_pair1, session_pair2], positions
            ):
                boxes = ax.boxplot(
                    similarities[session_pair][age],
                    positions=position,
                    patch_artist=True,
                )
                for patch, med in zip(boxes["boxes"], boxes["medians"]):
                    patch.set_facecolor(color)
                    med.set(color="k")

            if age == "aged":
                ax.set_yticks([])
            else:
                ax.set_ylabel("Cosine similarity of matched assemblies")
            ax.set_xticks(label_positions)
            ax.set_xticklabels(mice, rotation=45)

        return similarities

    def correlate_activation_location_to_cosine_similarity(
        self, session_pair, activation_thresh=1
    ):
        """
        For a given session pair and for each assembly, plot the cosine similarity against the displacement of the location
        where the assembly usually activates across the two sessions. This is good for assessing whether the pattern
        similarity is related to stability of the location of activation.

        :param session_pair:
        :param activation_thresh:
        :return:
        """
        assembly_matches = dict()
        best_similarities = dict()
        activation_locations = dict()
        activation_distances = dict()

        for mouse in self.meta["mice"]:
            registered_patterns = self.match_ensembles(mouse, session_pair)
            assembly_matches[mouse] = registered_patterns["matches"]
            best_similarities[mouse] = registered_patterns["best_similarities"]

            lin_positions = [
                self.data[mouse][session].behavior.data["df"]["lin_position"]
                for session in session_pair
            ]

            activation_locations[mouse] = np.zeros((2, len(assembly_matches[mouse])))

            activations = [
                self.data[mouse][session].assemblies["activations"]
                for session in session_pair
            ]

            for s1_assembly, s2_assembly in enumerate(assembly_matches[mouse]):
                above_threshold = (
                    zscore(activations[0][s1_assembly]) > activation_thresh
                )
                activation_locations[mouse][0, s1_assembly] = circmean(
                    lin_positions[0][above_threshold]
                )

                above_threshold = (
                    zscore(activations[1][s2_assembly]) > activation_thresh
                )
                activation_locations[mouse][1, s1_assembly] = circmean(
                    lin_positions[1][above_threshold]
                )

            activation_distances[mouse] = get_circular_error(
                activation_locations[mouse][0],
                activation_locations[mouse][1],
                np.pi * 2,
            )

        fig, ax = plt.subplots()
        for c, age in zip(["k", "r"], ["young", "aged"]):
            for mouse in self.meta["grouped_mice"][age]:
                ax.scatter(
                    best_similarities[mouse], activation_distances[mouse], color=c
                )
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("COM distance across sessions")
        ax = beautify_ax(ax)

        return (
            assembly_matches,
            best_similarities,
            activation_locations,
            activation_distances,
        )

    def pattern_similarity_matrix(self, mouse):
        n_sessions = len(self.meta["session_types"])
        best_similarities = np.zeros((n_sessions, n_sessions), dtype=object)
        best_similarities_median = nan_array((n_sessions, n_sessions))
        for i, s1 in enumerate(self.meta["session_types"]):
            for j, s2 in enumerate(self.meta["session_types"]):
                if i != j:
                    registered_ensembles = self.match_ensembles(mouse, (s1, s2))
                    best_similarities_this_pair = registered_ensembles[
                        "best_similarities"
                    ][~registered_ensembles["poor_matches"]]
                    best_similarities[i, j] = best_similarities_this_pair
                    best_similarities_median[i, j] = np.nanmedian(
                        best_similarities_this_pair
                    )

        return best_similarities, best_similarities_median

    def plot_pattern_similarity_matrix(self):
        """
        For each assembly in a given session, for another given session, find the assembly with the most similar
        pattern, assessed through cosine similarity. Do this for each assembly and each session pair. Take the median
        across assemblies for each session pair. Now you have a session x session matrix where each entry is the median
        cosine similarity of the best-matched assembly patterns across those two sessions. This could represent general
        similarity of assembly participation amongst cells. Do this for each mouse and then take the median across those
        medians. Also split into young versus aged.

        :return:
        """
        n_sessions = len(self.meta["session_types"])
        best_similarities = {
            age: nan_array(
                (len(self.meta["grouped_mice"][age]), n_sessions, n_sessions)
            )
            for age in ["young", "aged"]
        }
        for age in ["young", "aged"]:
            for i, mouse in tqdm(enumerate(self.meta["grouped_mice"][age])):
                best_similarities[age][i] = self.pattern_similarity_matrix(mouse)[1]

        vmin = min(
            [
                np.nanmin(np.nanmedian(best_similarities[age], axis=0))
                for age in ["aged", "young"]
            ]
        )
        vmax = max(
            [
                np.nanmax(np.nanmedian(best_similarities[age], axis=0))
                for age in ["aged", "young"]
            ]
        )

        fig, axs = plt.subplots(1, 2, figsize=(12.6, 6.5))
        for ax, age in zip(axs, ["young", "aged"]):
            ax.imshow(
                np.nanmedian(best_similarities[age], axis=0),
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            ax.set_title(age)
            ax.set_xticks(range(n_sessions))
            ax.set_yticks(range(n_sessions))
            ax.set_xticklabels(self.meta["session_types"], rotation=45)
            ax.set_yticklabels(self.meta["session_types"])

    def make_ensemble_fields(
        self,
        mouse,
        session_type,
        spatial_bin_size_radians=0.05,
        running_only=False,
        std_thresh=2,
        get_zSI=False,
    ):
        """
        Make a single Pastalkova (snake) plot depicting z-scored assembly activation strength as a function of spatial
        location.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_type: str
            Session name (Goals1, 2, 3, 4, or Reversal).

        spatial_bin_size: float
            Spatial bin size in radians. Should be the same as the one run with PlaceFields() for consistency.

        show_plot: bool
            Whether to plot.

        ax: Axis object.
            Axis to plot on.

        order: None or (assembly,) array
            Order to plot assemblies in. If order is None and do_sort is False, leave it alone. If order has a value,
            overrides do_sort.

        do_sort: bool
            Whether to sort assembly order based on peak location.
        """
        ensembles = self.data[mouse][session_type].assemblies
        behavior_data = self.data[mouse][session_type].behavior.data

        if std_thresh is None:
            activations = ensembles["activations"]
        else:
            stds = np.std(ensembles["activations"], axis=1)
            means = np.mean(ensembles["activations"], axis=1)
            thresh = means + std_thresh * stds
            activations = ensembles["activations"].copy()
            activations[activations < np.tile(thresh, [activations.shape[1], 1]).T] = 0

        placefield_bin_size = self.data[mouse][session_type].meta["spatial_bin_size"]
        if spatial_bin_size_radians != placefield_bin_size:
            warnings.warn(
                f"Spatial bin size does not match PlaceField class's value of {placefield_bin_size}"
            )

        if running_only:
            velocity_threshold = self.data[mouse][session_type].meta["threshold"]
        else:
            velocity_threshold = 0

        ensemble_fields = PlaceFields(
            np.asarray(behavior_data["df"]["t"]),
            np.asarray(behavior_data["df"]["x"]),
            np.asarray(behavior_data["df"]["y"]),
            activations,
            bin_size=spatial_bin_size_radians,
            circular=True,
            fps=self.data[mouse][session_type].behavior.meta["fps"],
            shuffle_test=get_zSI,
            velocity_threshold=velocity_threshold,
        )

        return ensemble_fields

    def find_ensemble_dist_to_reward(self, mouse, session_type):
        fields = self.data[mouse][session_type].assemblies["fields"]
        behavior_data = self.data[mouse][session_type].behavior.data
        lin_position = behavior_data["df"]["lin_position"]
        port_locations = np.asarray(behavior_data["lin_ports"])[
            behavior_data["rewarded_ports"]
        ]
        spatial_bin_size_radians = fields.meta["bin_size"]

        reward_location_bins, bins = find_reward_spatial_bins(
            lin_position,
            port_locations,
            spatial_bin_size_radians=spatial_bin_size_radians,
        )

        d = np.min(
            np.hstack(
                [
                    [
                        get_circular_error(center, reward_bin, len(bins))
                        for center in fields.data["placefield_centers"]
                    ]
                    for reward_bin in reward_location_bins
                ]
            ),
            axis=1,
        )

        return d

    def plot_ensemble_field_distances_to_rewards(self):
        """
        For each session, find  the distances of all ensemble fields to
        the closest reward location. Compare young vs aged.

        :return:
        """
        d = dict()
        for session in self.meta["session_types"]:
            d[session] = dict()

            for age in ages:
                d[session][age] = []

                for mouse in self.meta["grouped_mice"][age]:
                    d[session][age].extend(
                        self.find_ensemble_dist_to_reward(mouse, session)
                    )

        fig, axs = plt.subplots(1, len(self.meta["session_types"]))
        fig.subplots_adjust(wspace=0)
        for session, ax in zip(self.meta["session_types"], axs):
            boxes = ax.boxplot(
                [d[session]["young"], d[session]["aged"]],
                widths=0.75,
                patch_artist=True,
            )

            for patch, med, color in zip(boxes["boxes"], boxes["medians"], age_colors):
                patch.set_facecolor(color)
                med.set(color="k")

            if session == "Goals1":
                ax.set_ylabel("Median distance of field center to reward")
            else:
                ax.set_yticks([])
            ax.set_xticklabels(ages, rotation=45)

        return d

    def snakeplot_ensembles(
        self,
        mouse,
        session_type,
        spatial_bin_size_radians=0.05,
        show_plot=True,
        ax=None,
        order=None,
        do_sort=True,
    ):
        # Get ensemble fields and relevant behavior data such as linearized position and port location.
        ensemble_fields = self.make_ensemble_fields(
            mouse, session_type, spatial_bin_size_radians=spatial_bin_size_radians
        )
        behavior_data = self.data[mouse][session_type].behavior.data
        lin_position = behavior_data["df"]["lin_position"]
        port_locations = np.asarray(behavior_data["lin_ports"])[
            behavior_data["rewarded_ports"]
        ]

        # Determine order of assemblies.
        if order is None and do_sort:
            order = np.argsort(ensemble_fields.data["placefield_centers"])
        elif order is None and not do_sort:
            order = range(ensemble_fields.data["placefields_normalized"].shape[0])

        # Plot assemblies.
        if show_plot:
            # Convert port locations to bin #.
            reward_locations_bins = find_reward_spatial_bins(
                lin_position,
                port_locations,
                spatial_bin_size_radians=spatial_bin_size_radians,
            )[0]

            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5.5))
            ax.imshow(ensemble_fields.data["placefields_normalized"][order])
            ax.axis("tight")
            ax.set_ylabel("Ensemble #")
            ax.set_xlabel("Location")
            ax.set_title(f"{mouse}, {session_type}")

            for port in reward_locations_bins:
                ax.axvline(port, c="g")

        return ensemble_fields

    def ensemble_membership_changes(self, mouse, session_types):
        # Register the ensembles and for both sessions, find members of each ensemble.
        registered_ensembles = self.match_ensembles(mouse, session_types)

        # Find the highest contributing neurons for each from both
        # sessions. This gives you a few neuron indices for each ensemble
        # and each session, referenced to the "matched_patterns" matrix,
        # which is after pruning away non-registered neurons. These
        # are NOT the neuron indices referenced to single sessions.
        members_ = {
            session_type: find_members(
                registered_ensembles["matched_patterns"][session_number]
            )[1]
            for session_number, session_type in enumerate(session_types)
        }

        # We also want the neuron indices referenced to single sessions
        # for convenience on a number of functions. Get the registration
        # mappings so that we can use the "pruned" indices to retrieve
        # the within-session indices.
        trimmed_map = np.asarray(self.get_cellreg_mappings(mouse, session_types)[0])

        # Same as members_, but referenced within a session (without
        # pruning non-registered neurons.
        members = {
            session_type: [
                trimmed_map[ensemble_members, i]
                for ensemble_members in members_[session_type]
            ]
            for i, session_type in enumerate(session_types)
        }

        # Find neurons that were consistent across both sessions, meaning
        # they were members in both. Referenced to pruned patterns.
        consistent_members_ = [
            np.intersect1d(a, b)
            for a, b in zip(members_[session_types[0]], members_[session_types[1]])
        ]

        # Same as above but referenced to single sessions.
        consistent_members = {
            session_type: [
                trimmed_map[consistent_members_this_ensemble, i]
                for consistent_members_this_ensemble in consistent_members_
            ]
            for i, session_type in enumerate(session_types)
        }

        dropouts_ = []
        newcomers_ = []
        for members_s1, members_s2 in zip(
            members_[session_types[0]], members_[session_types[1]]
        ):
            members_s1 = np.asarray(members_s1)
            members_s2 = np.asarray(members_s2)
            dropouts_.append(members_s1[~np.isin(members_s1, members_s2)])
            newcomers_.append(members_s2[~np.isin(members_s2, members_s1)])

        dropouts = {
            session_type: [
                trimmed_map[dropouts_this_session, i]
                for dropouts_this_session in dropouts_
            ]
            for i, session_type in enumerate(session_types)
        }

        newcomers = {
            session_type: [
                trimmed_map[newcomers_this_session, i]
                for newcomers_this_session in newcomers_
            ]
            for i, session_type in enumerate(session_types)
        }

        return members, consistent_members, dropouts, newcomers

    def plot_average_ensemble_member_field(
        self, mouse, session_type, ensemble_number, S_type="S"
    ):
        """
        Finds the aggregate or average firing field of an ensemble member's field.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_type: str
            Session name (e.g. 'Goals4').

        ensemble_number: int
            Specify an ensemble index.

        S_type: str
            'S' or 'S_binary'.
        """
        ensembles = self.data[mouse][session_type].assemblies
        spatial_data = self.data[mouse][session_type].spatial
        imaging_data = self.data[mouse][session_type].imaging[S_type]
        behavior_data = self.data[mouse][session_type].behavior.data

        ports = find_reward_spatial_bins(
            spatial_data.data["x"],
            np.asarray(behavior_data["lin_ports"])[behavior_data["rewarded_ports"]],
            spatial_bin_size_radians=spatial_data.meta["bin_size"],
        )[0]
        members = find_members(ensembles["patterns"][ensemble_number])[1]
        fields = np.vstack(
            [
                spatial_bin(
                    spatial_data.data["x"],
                    spatial_data.data["y"],
                    bin_size_cm=spatial_data.meta["bin_size"],
                    weights=neuron,
                    one_dim=True,
                    bins=spatial_data.data["occupancy_bins"],
                )[0]
                for neuron in imaging_data
            ]
        )

        fig, ax = plt.subplots()
        normalized_field = np.mean(fields[members], axis=0) / np.mean(fields, axis=0)
        ax.plot(normalized_field)
        [ax.axvline(x=port, color="r") for port in ports]

    def get_spatial_ensembles(self, mouse, session_type, pval_threshold=0.05):
        is_spatial = (
            self.data[mouse][session_type]
            .assemblies["fields"]
            .data["spatial_info_pvals"]
            < pval_threshold
        )

        return is_spatial

    def find_proportion_spatial_ensembles(self, pval_threshold=0.05):
        p = dict()
        for session in self.meta["session_types"]:
            p[session] = dict()

            for age in ages:
                p[session][age] = []

                for mouse in self.meta["grouped_mice"][age]:
                    is_spatial = self.get_spatial_ensembles(
                        mouse, session, pval_threshold
                    )

                    p[session][age].append(sum(is_spatial) / len(is_spatial))

        fig, axs = plt.subplots(1, len(self.meta["session_types"]))
        fig.subplots_adjust(wspace=0)

        for i, (ax, session) in enumerate(zip(axs, self.meta["session_types"])):
            box = ax.boxplot(
                [p[session][age] for age in ages],
                labels=ages,
                patch_artist=True,
                widths=0.75,
            )
            for patch, med, color in zip(box["boxes"], box["medians"], age_colors):
                patch.set_facecolor(color)
                med.set(color="k")
            ax.set_xticks([])
            ax.set_title(session)

            if i > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel("Proportion of ensembles with spatial selectivity")

        patches = [
            mpatches.Patch(facecolor=c, label=label, edgecolor="k")
            for c, label in zip(age_colors, ["Young", "Aged"])
        ]
        fig.legend(handles=patches, loc="lower right")

        return p

    def map_ensemble_fields(
        self,
        mouse,
        session_pair,
        spatial_bin_size_radians=0.05,
        running_only=False,
        std_thresh=2,
        reference_session=0,
    ):
        """
        Maps two sessions' ensembles.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_pair: tuple
            Pair of sessions you want to register ensembles from.

        spatial_bin_size_radians: don't change this

        running_only: bool
            Whether to only consider bins where the mouse is running.

        std_thresh: int
            Number of standard deviations above the mean for an ensemble to be considered active.

        reference_session: 0 or 1
            Which session of the pair to be considered the reference. match_ensembles() will take ensembles
            from the first session and register them to the second session. Do this functionally by flipping the
            tuple values.

        :returns
        ---
        ensemble_fields: list of (n_ensembles, spatial_bins) ensemble spatial fields.
        ensemble_field_data: list of PlaceFields class instances.

        """
        if reference_session == 0:
            pass
        elif reference_session == 1:
            session_pair = (session_pair[1], session_pair[0])
        else:
            raise ValueError(f'Invalid value for reference_session: {reference_session}')

        # Register the ensembles.
        registered_ensembles = self.match_ensembles(mouse, session_pair)

        ensemble_field_data = [self.make_ensemble_fields(
            mouse,
            session,
            spatial_bin_size_radians=spatial_bin_size_radians,
            running_only=running_only,
            std_thresh=std_thresh,
            get_zSI=False,
        ) for i, session in enumerate(session_pair)]

        # Get the spatial fields of the ensembles and reorder them.
        ensemble_fields = [
            ensemble_field_data[0].data["placefields_normalized"],
            ensemble_field_data[1].data["placefields_normalized"][
                registered_ensembles["matches"]
            ],
        ]
        ensemble_fields[1][registered_ensembles["poor_matches"]] = np.nan

        # If the second session was the reference, we had swapped the positions of the sessions prior
        # to registration, so the outputs of match_ensembles() don't correspond to the original session_pair
        # input. Swap the list items so they do.
        if reference_session == 1:
            ensemble_fields[0], ensemble_fields[1] = ensemble_fields[1], ensemble_fields[0]
            ensemble_field_data[0], ensemble_field_data[1] = ensemble_field_data[1], ensemble_field_data[1]

        return ensemble_fields, ensemble_field_data

    def correlate_ensemble_fields(
        self, mouse, session_types, spatial_bin_size_radians=0.05
    ):
        ensemble_fields = self.map_ensemble_fields(
            mouse, session_types, spatial_bin_size_radians=spatial_bin_size_radians
        )[0]

        rhos = [
            spearmanr(ensemble_day1, ensemble_day2)[0]
            for ensemble_day1, ensemble_day2 in zip(
                ensemble_fields[0], ensemble_fields[1]
            )
        ]

        return np.asarray(rhos)

    def plot_ensemble_field_correlations(self, session_pair: tuple, show_plot=True):
        ensemble_field_rhos = dict()
        for age in ages:
            ensemble_field_rhos[age] = []

            for mouse in self.meta["grouped_mice"][age]:
                rhos = self.correlate_ensemble_fields(mouse, session_pair)
                ensemble_field_rhos[age].append(rhos[~np.isnan(rhos)])

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, color, ax in zip(ages, age_colors, axs):
                boxes = ax.boxplot(ensemble_field_rhos[age], patch_artist=True)

                for patch, med in zip(boxes["boxes"], boxes["medians"]):
                    patch.set_facecolor(color)
                    med.set(color="k")

                if age == "aged":
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("Spearman rho of matched ensemble fields")
                ax.set_xticklabels(self.meta["grouped_mice"][age], rotation=45)

        return ensemble_field_rhos

    def plot_ensemble_field_correlation_comparisons(
        self,
        session_pair1=("Goals3", "Goals4"),
        session_pair2=("Goals4", "Reversal"),
        show_plot=True,
    ):
        ensemble_field_rhos = dict()
        for session_pair in [session_pair1, session_pair2]:
            ensemble_field_rhos[session_pair] = self.plot_ensemble_field_correlations(
                session_pair, show_plot=False
            )

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, ax, color in zip(ages, axs, age_colors):
                mice = self.meta["grouped_mice"][age]
                positions = [
                    np.arange(start, start + 3 * len(mice), 3) for start in [-0.5, 0.5]
                ]
                label_positions = np.arange(0, 3 * len(mice), 3)

                for session_pair, position in zip(
                    [session_pair1, session_pair2], positions
                ):
                    boxes = ax.boxplot(
                        ensemble_field_rhos[session_pair][age],
                        positions=position,
                        patch_artist=True,
                    )
                    for patch, med in zip(boxes["boxes"], boxes["medians"]):
                        patch.set_facecolor(color)
                        med.set(color="k")

                if age == "aged":
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("Spatial correlation of matched assemblies")
                ax.set_xticks(label_positions)
                ax.set_xticklabels(mice, rotation=45)

        return ensemble_field_rhos

    def correlate_ensemble_fields_across_ages(self, session_pair, ax=None):
        """
        For a pair of sessions, correlate each registered ensemble per mouse.
        Take the median across ensembles for each mouse and group them into
        either young or aged.

        :param session_pair:
        :param ax:
        :return:
        """
        rhos = {
            age: [
                np.nanmedian(self.correlate_ensemble_fields(mouse, session_pair))
                for mouse in self.meta["grouped_mice"][age]
            ]
            for age in ages
        }

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 5))
        boxes = ax.boxplot([rhos[age] for age in ages], widths=0.75, patch_artist=True)
        for patch, med, color in zip(boxes["boxes"], boxes["medians"], age_colors):
            patch.set_facecolor(color)
            med.set(color="k")
        ax.set_xticklabels(ages)
        ax.set_title(session_pair)
        ax.set_ylabel("Ensemble field correlation [Spearman rho]")
        plt.tight_layout()

        return rhos

    def snakeplot_matched_ensembles(
        self,
        mouse,
        session_types,
        spatial_bin_size_radians=0.05,
        show_plot=True,
        axs=None,
        sort_by=0,
        subset=None,
        reference_session=0,
    ):
        # Map the ensembles to each other.
        ensemble_fields = self.map_ensemble_fields(
            mouse, session_types, spatial_bin_size_radians=spatial_bin_size_radians,
            reference_session=reference_session,
        )[0]

        # If no subset was specified, take all the ensembles.
        if subset is None:
            subset = range(len(ensemble_fields[0]))

        # Get linearized position and port locations.
        lin_positions = [
            self.data[mouse][session].behavior.data["df"]["lin_position"]
            for session in session_types
        ]
        port_locations = [
            np.asarray(self.data[mouse][session].behavior.data["lin_ports"])[
                self.data[mouse][session].behavior.data["rewarded_ports"]
            ]
            for session in session_types
        ]

        if spatial_bin_size_radians is None:
            spatial_bin_size_radians = [
                self.data[mouse][session].spatial.meta["bin_size"]
                for session in session_types
            ]
            assert (
                len(np.unique(spatial_bin_size_radians)) == 1
            ), "Different bin sizes in two sessions."
            spatial_bin_size_radians = spatial_bin_size_radians[0]

        # Convert port locations to bin #.
        port_locations_bins = [find_reward_spatial_bins(lin_position, port_location,
                                                       spatial_bin_size_radians)[0]
                               for lin_position, port_location in zip(lin_positions, port_locations)]

        # Sort the fields.
        order = np.argsort(np.argmax(ensemble_fields[sort_by], axis=1))

        # Plot.
        if show_plot:
            session_labels = [session.replace('Goals', 'Training') for session in session_types]
            if axs is None:
                fig, axs = plt.subplots(1, len(session_types), figsize=(10, 7))

            for ax, fields, ports, session in zip(
                axs, ensemble_fields, port_locations_bins, session_labels
            ):
                ax.imshow(fields[order][subset])
                ax.axis("tight")
                ax.set_ylabel("Ensemble #")
                ax.set_xlabel("Location")
                ax.set_title(session)

                for port in ports:
                    ax.axvline(port, c="r", alpha=0.5)
            fig.suptitle(mouse)

        return ensemble_fields

    def snakeplot_matched_fading_ensembles(self):
        for mouse in self.meta['mice']:
            ensemble_trends = self.find_activity_trends(mouse, 'Reversal')[0]
            self.snakeplot_matched_ensembles(mouse, ('Goals4', 'Reversal'), sort_by=1,
                                             reference_session=1, subset=ensemble_trends['decreasing'])

    def boxplot_assembly_SI(self, session_type):
        SI = {
            age: [
                self.data[mouse][session_type]
                .assemblies["fields"]
                .data["spatial_info_z"]
                for mouse in self.meta["grouped_mice"][age]
            ]
            for age in ages
        }

        fig, axs = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0)

        for age, ax, color in zip(ages, axs, age_colors):
            mice = self.meta["grouped_mice"][age]
            boxes = ax.boxplot(SI[age], patch_artist=True)

            # color the boxplots.
            for patch, med in zip(boxes["boxes"], boxes["medians"]):
                patch.set_facecolor(color)
                med.set(color="k")

            if age == "aged":
                ax.set_yticks([])
            else:
                ax.set_ylabel("Assembly spatial information (z)")
            ax.set_xticklabels(mice, rotation=45)

        return SI

    def boxplot_all_assembly_SI(self):
        SI = {
            session_type: {
                age: [
                    np.mean(
                        self.data[mouse][session_type]
                        .assemblies["fields"]
                        .data["spatial_info_z"]
                    )
                    for mouse in self.meta["grouped_mice"][age]
                ]
                for age in ages
            }
            for session_type in self.meta["session_types"]
        }

        fig, axs = plt.subplots(1, len(self.meta["session_types"]))
        fig.subplots_adjust(wspace=0)

        for ax, session_type, title in zip(
            axs, self.meta["session_types"], self.meta["session_labels"]
        ):
            boxes = ax.boxplot(
                [SI[session_type][age] for age in ages], patch_artist=True, widths=0.75
            )

            for patch, med, color in zip(boxes["boxes"], boxes["medians"], age_colors):
                patch.set_facecolor(color)
                med.set(color="k")

            if session_type != self.meta["session_types"][0]:
                ax.set_yticks([])
            else:
                ax.set_ylabel("Mean assembly spatial information (z)")
            ax.set_xticks([])
            ax.set_title(title)

        return SI

    # def correlate_stability_to_reversal(
    #     self,
    #     corr_sessions=("Goals3", "Goals4"),
    #     criterion_session="Reversal",
    # ):
    #     median_r = []
    #     criterion = []
    #     for mouse in self.meta["mice"]:
    #         corrs = self.correlate_fields(mouse, corr_sessions, show_histogram=False)
    #         median_r.append(np.nanmedian(corrs["r"]))
    #         behavior = self.data[mouse][criterion_session].behavior
    #         criterion.append(behavior.data["learning"]["criterion"])
    #
    #     fig, ax = plt.subplots()
    #     ax.scatter(median_r, criterion)
    #     ax.set_xlabel("Median place field correlation Goals1 vs Goals 2 [r]")
    #     ax.set_ylabel("Trials to criterion Reversal1")
    #
    #     return median_r, criterion


if __name__ == "__main__":
    RR = RecentReversal(
        [
            "Fornax",
            "Gemini",
            # "Io",
            "Janus",
            "Lyra",
            "Miranda",
            "Naiad",
            "Oberon",
            "Puck",
            # "Rhea",
            "Sao",
            "Titania",
            "Umbriel",
            "Virgo",
            "Ymir",
            "Atlas",
        ],
        project_name="RemoteReversal",
        behavior_only=False,
    )
    n_ensembles = RR.count_ensembles(normalize=False)
    n_ensembles_norm = RR.count_ensembles(normalize=True)
    RR.data["Fornax"]["Goals4"].plot_assembly(0)

    # Only Goals2 p = 0.03.
    RR.plot_ensemble_sizes(data_type="ensemble_size")

    # Goals1, Goals2 p = 0.04, Goals3, Goals4 p = 0.07
    ensemble_makeup = RR.plot_ensemble_sizes(data_type="unique_members")

    # Goals2 p = 0.02, Goals1, Goals3 p = 0.07. Aged mice catch up slowly?
    SI = RR.boxplot_all_assembly_SI()

    RR.snakeplot_matched_ensembles("Fornax", ("Goals3", "Goals4"))
    RR.snakeplot_matched_ensembles("Fornax", ("Goals4", "Reversal"))

    p_matches = RR.percent_matched_ensembles()
    reg_similarities = RR.plot_pattern_cosine_similarity_comparisons()
    ensemble_field_rhos = RR.plot_ensemble_field_correlation_comparisons()

    pass
