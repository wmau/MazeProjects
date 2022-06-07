import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cm
from CaImaging.util import (
    sem,
    nan_array,
    bin_transients,
    make_bins,
    ScrollPlot,
    contiguous_regions,
    stack_padding,
    distinct_colors,
    group_consecutives,
    open_minian,
    cluster_corr,
)
import seaborn as sns
import math
import random
import ruptures as rpt
import networkx as nx
from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from CaImaging.plotting import (
    errorfill,
    beautify_ax,
    jitter_x,
    shiftedColorMap,
    plot_xy_line,
)
from scipy.stats import (
    spearmanr,
    zscore,
    circmean,
    kendalltau,
    wilcoxon,
    mannwhitneyu,
    pearsonr,
    ttest_ind,
    chisquare,
    ttest_rel,
    ttest_1samp,
    kstest,
    uniform,
)
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from scipy.spatial import distance
from joblib import Parallel, delayed
from CircleTrack.SessionCollation import MultiAnimal
from CircleTrack.MiniscopeFunctions import CalciumSession
from CaImaging.CellReg import rearrange_neurons, trim_map, scrollplot_footprints
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import StratifiedKFold, KFold
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from CircleTrack.plotting import (
    plot_daily_rasters,
    spiral_plot,
    highlight_column,
    plot_port_activations,
    color_boxes,
    plot_raster,
)
from CaImaging.Assemblies import (
    find_assemblies,
    preprocess_multiple_sessions,
    lapsed_activation,
)
from CircleTrack.Assemblies import (
    plot_assembly,
    find_members,
    find_memberships,
    plot_pattern,
)
import xarray as xr
import pymannkendall as mk
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product, cycle, islice
from CaImaging.PlaceFields import spatial_bin, PlaceFields, define_field_bins
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
plt.rcParams.update({"font.size": 16})

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

ages = ["young", "aged"]
age_colors = ["turquoise", "royalblue"]

class RecentReversal:
    def __init__(
        self,
        mice,
        project_name="RemoteReversal",
        behavior_only=False,
        save_figs=True,
        ext="pdf",
        save_path=r"C:\Users\wm228\Documents\GitHub\memory_flexibility\Figures",
    ):
        # Collect data from all mice and sessions.
        self.data = MultiAnimal(
            mice,
            project_name=project_name,
            SessionFunction=CalciumSession,
        )

        self.save_configs = {
            "save_figs": save_figs,
            "ext": ext,
            "path": save_path,
        }

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
    def save_fig(self, fig, fname, folder):
        fpath = os.path.join(
            self.save_configs["path"],
            str(folder),
            f'{fname}.{self.save_configs["ext"]}',
        )
        fig.savefig(fpath, bbox_inches="tight")

    def ages_to_plot_parser(self, ages_to_plot):
        if ages_to_plot is None:
            ages_to_plot = ages
            plot_colors = age_colors
        else:
            plot_colors = [age_colors[ages.index(ages_to_plot)]]
        if type(ages_to_plot) is str:
            ages_to_plot = [ages_to_plot]

        n_ages_to_plot = len(ages_to_plot)

        return ages_to_plot, plot_colors, n_ages_to_plot

    def rearrange_neurons(
        self, mouse, selected_sessions, data_type, detected="everyday"
    ):
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
            mouse, selected_sessions, detected=detected, neurons_from_session1=None
        )[0]

        # Get calcium activity from each session for this mouse.
        if data_type == "patterns":
            activity_list = [
                sessions[session].assemblies[data_type].T
                for session in selected_sessions
            ]
        else:
            activity_list = [
                sessions[session].imaging[data_type] for session in selected_sessions
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

    def set_age_legend(self, fig, loc="lower right"):
        patches = [
            mpatches.Patch(
                facecolor=c, label=label.replace("aged", "middle-aged"), edgecolor="k"
            )
            for c, label in zip(age_colors, ages)
        ]
        fig.legend(handles=patches, loc=loc)

    def plot_neuron_count(self, sessions_to_plot=None, ages_to_plot=None):
        """
        Plot the number of neurons in each session.

        :param sessions_to_plot:
        :return:
        """
        # Get sessions and labels.
        if sessions_to_plot is None:
            sessions_to_plot = self.meta["session_types"]
        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in sessions_to_plot
        ]

        # Get number of neurons.
        n_neurons = pd.DataFrame(index=self.meta["mice"])
        for session in sessions_to_plot:
            n_neurons[session] = [
                self.data[mouse][session].imaging["n_neurons"]
                for mouse in self.meta["mice"]
            ]

        n_neurons["aged"] = [self.meta["aged"][mouse] for mouse in n_neurons.index]

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        fig, ax = plt.subplots()
        for age, color in zip(ages_to_plot, plot_colors):
            aged = True if age == "aged" else False
            n_neurons_plot = n_neurons.loc[n_neurons.aged == aged]
            n_neurons_plot = n_neurons_plot.drop(columns="aged")
            xticks = [
                col.replace("Goals", "Training") for col in n_neurons_plot.columns
            ]
            ax.plot(xticks, n_neurons_plot.T, color=color, alpha=0.5)
            errorfill(
                xticks,
                n_neurons_plot.mean(axis=0),
                yerr=sem(n_neurons_plot, axis=0),
                ax=ax,
                color=color,
            )
            plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_ylim([0, 1500]) # Hard coded for Denise's aesthetic preference.
        ax.set_ylabel("# neurons", fontsize=22)
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        # self.set_age_legend(fig)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0)

        n_neurons = n_neurons.sort_values(by="aged")
        return n_neurons, fig

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

    def behavior_over_trials(
        self,
        session_type,
        window=6,
        strides=2,
        performance_metric="CRs",
        trial_limit=None,
    ):
        dv, mice, sessions, groups, trial_blocks = [], [], [], [], []
        for age in ages:
            for mouse in self.meta["grouped_mice"][age]:
                session = self.data[mouse][session_type].behavior
                session.sdt_trials(
                    rolling_window=window,
                    trial_interval=strides,
                    plot=False,
                    trial_limit=trial_limit,
                )

                n = range(len(session.sdt[performance_metric]))
                dv.extend(session.sdt[performance_metric])
                mice.extend([mouse for i in n])
                sessions.extend([session_type for i in n])
                groups.extend([age for i in n])
                trial_blocks.extend([i for i in n])

        df = pd.DataFrame(
            {
                "t": trial_blocks,
                "dv": dv,
                "mice": mice,
                "session": sessions,
                "age": groups,
            }
        )

        # cols = ['mice', 'session', 'group']
        # df[cols] = df[cols].mask(df[cols]=='nan', None).ffill(axis=0)
        # df['t'] = df['t'].isna().cumsum() + df['t'].ffill()
        # df = df.dropna(axis=0)

        return df

    def stack_behavior_dv(self, df):
        dv = dict()

        for age in ages:
            dv_temp = []

            for mouse in self.meta["grouped_mice"][age]:
                dv_temp.append(df.loc[df["mice"] == mouse, "dv"])
                dv[age] = stack_padding(dv_temp)

        return dv

    def plot_trial_behavior(
        self,
        ages_to_plot=None,
        session_types=None,
        performance_metric="d_prime",
        plot_sig=False,
        **kwargs,
    ):
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        if session_types is None:
            session_types = self.meta["session_types"]
        n_sessions = len(session_types)

        dv, pvals, anova_dfs = dict(), dict(), dict()
        sessions_df = pd.DataFrame()
        for session_type in session_types:
            anova_dfs[session_type], df, pvals_temp = self.trial_behavior_anova(
                session_type, performance_metric=performance_metric, **kwargs
            )
            dv[session_type] = self.stack_behavior_dv(df)
            pvals[session_type] = pvals_temp

            sessions_df = pd.concat((sessions_df, df))

        ylabel = {
            "d_prime": "d'",
            "CRs": "Correct rejection rate",
            "hits": "Hit rate",
        }
        if n_sessions == 1:
            fig, axs = plt.subplots(1, n_sessions, figsize=(5, 5))
            axs = [axs]
        else:
            fig, axs = plt.subplots(
                1, n_sessions, figsize=(5 * n_sessions, 5), sharey=True
            )
        for i, (ax, session_type) in enumerate(zip(axs, session_types)):
            for age, color in zip(ages_to_plot, plot_colors):
                y = dv[session_type][age]
                xrange = y.shape[1]
                ax.plot(range(xrange), y.T, color=color, alpha=0.3)
                errorfill(
                    range(xrange),
                    np.nanmean(y, axis=0),
                    sem(y, axis=0),
                    ax=ax,
                    color=color,
                    label=age,
                )
            xlims = [int(x) for x in ax.get_xlim()]
            ax.set_xticks(xlims)
            n_trials = np.max(
                [
                    self.data[mouse][session_type].behavior.data["ntrials"]
                    for mouse in self.meta["mice"]
                ]
            )
            ax.set_xticklabels([1, n_trials])

            ax.set_title(session_type.replace("Goals", "Training"))

            if i == 0:
                ax.set_ylabel(ylabel[performance_metric])
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]

            if plot_sig:
                sig = np.where(pvals[session_type] < 0.05)[0]
                if len(sig):
                    sig_regions = group_consecutives(sig, step=1)
                    ylims = ax.get_ylim()
                    for region in sig_regions:
                        ax.fill_between(
                            np.arange(region[0], region[-1]),
                            ylims[-1],
                            ylims[0],
                            alpha=0.4,
                            color="gray",
                        )

        if n_sessions == 1:
            axs[0].set_xlabel("Trials")
        else:
            fig.supxlabel("Trials")
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        axs[-1].legend(loc="lower right")

        return dv, anova_dfs, fig, sessions_df

    def trial_behavior_anova(
        self, session_type, performance_metric="d_prime", **kwargs
    ):
        df = self.behavior_over_trials(
            session_type, performance_metric=performance_metric, **kwargs
        )

        anova_df = pg.anova(df, dv="dv", between=["t", "age"], ss_type=1)

        pvals = []
        for i in range(np.max(df["t"])):
            x = df.loc[np.logical_and(df["age"] == "young", df["t"] == i), "dv"]
            y = df.loc[np.logical_and(df["age"] == "aged", df["t"] == i), "dv"]

            pval = ttest_ind(x, y, nan_policy="omit").pvalue
            if ~np.isnan(pval):
                pvals.append(pval)

        pvals = multipletests(pvals, method="fdr_bh")[1]

        return anova_df, df, pvals

    def plot_reversal_vs_training4_trial_behavior(
        self, age, performance_metric="d_prime", **kwargs
    ):
        dv, pvals = dict(), dict()
        session_types = ("Goals4", "Reversal")
        for session_type in session_types:
            df = self.trial_behavior_anova(
                session_type, performance_metric=performance_metric, **kwargs
            )[1]
            dv[session_type] = self.stack_behavior_dv(df)

        # For significance markers.
        pvals = []
        for x, y in zip(dv[session_types[0]][age].T, dv[session_types[1]][age].T):
            pval = ttest_ind(x, y, nan_policy="omit").pvalue

            if np.isfinite(pval) and ((len(x) + len(y)) > 5):
                pvals.append(pval)
        pvals = multipletests(pvals, method="fdr_bh")[1]

        ylabel = {
            "d_prime": "d'",
            "CRs": "Correct rejection rate",
            "hits": "Hit rate",
        }
        fig, ax = plt.subplots(figsize=(5, 5))
        n_trials = []
        for session_type, color in zip(session_types, ["k", "cornflowerblue"]):
            y = dv[session_type][age]
            x = y.shape[1]
            ax.plot(range(x), y.T, color=color, alpha=0.3)
            errorfill(
                range(x),
                np.nanmean(y, axis=0),
                sem(y, axis=0),
                ax=ax,
                color=color,
                label=session_type.replace("Goals", "Training"),
            )

            n_trials.append(
                np.max(
                    [
                        self.data[mouse][session_type].behavior.data["ntrials"]
                        for mouse in self.meta["grouped_mice"][age]
                    ]
                )
            )

        sig = np.where(pvals < 0.05)[0]
        if len(sig):
            sig_regions = group_consecutives(sig, step=1)
            ylims = ax.get_ylim()
            for region in sig_regions:
                ax.fill_between(
                    np.arange(region[0], region[-1]),
                    ylims[-1],
                    ylims[0],
                    alpha=0.4,
                    color="gray",
                )

        ax.legend(loc="lower right", fontsize=14)
        ax.set_ylabel(ylabel[performance_metric])
        ax.set_xlabel("Trials")
        ax.set_xticks([int(x) for x in ax.get_xlim()])
        ax.set_xticklabels([1, np.max(n_trials)])
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        return dv, fig

    def learning_rate(
        self,
        session_type,
        window=6,
        strides=2,
        performance_metric="d_prime",
        trial_limit=None,
    ):
        df = self.behavior_over_trials(
            session_type,
            window=window,
            strides=strides,
            performance_metric=performance_metric,
            trial_limit=trial_limit,
        )

        def objective(x, a, b, c):
            return a * x + b * x ** 2 + c

        slopes, peak_trial, norm_peak_trial = [[] for i in range(3)]
        for mouse in df.mice.unique():
            mouse_df = df.loc[df["mice"] == mouse]
            X = mouse_df["t"].values
            y = mouse_df["dv"].values

            slopes.append(
                LinearRegression()
                .fit(np.reshape(X, (-1, 1)), np.reshape(y, (-1, 1)))
                .coef_[0][0]
            )

            curve_params = curve_fit(objective, X, y)[0]
            curve = objective(X, *curve_params)
            peak_trial.append(np.argmax(curve))
            norm_peak_trial.append(np.argmax(curve) / np.max(X))

            # ax.plot(X,y, 'b-')
            # ax.plot(X, objective(X, *p), 'r-')
        df = pd.DataFrame(
            {
                "mice": df.mice.unique(),
                "reg_slope": slopes,
                "peak_trial": peak_trial,
                "norm_peak_trial": norm_peak_trial,
            }
        )
        return df

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

    def aggregate_behavior_over_trials(
        self,
        window=6,
        strides=2,
        performance_metric="d_prime",
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
        df = self.aggregate_behavior_over_trials(
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

    def performance_session_type(
        self,
        session_type,
        window=None,
        strides=None,
        performance_metric="d_prime",
        downsample_trials=False,
    ):
        if downsample_trials:
            trial_limit = min(
                [
                    self.data[mouse][session_type].behavior.data["ntrials"]
                    for mouse in self.meta["mice"]
                ]
            )
        else:
            trial_limit = None

        behavioral_performance = self.aggregate_behavior_over_trials(
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

        return peak_performance

    def plot_performance_session_type(
        self,
        session_type,
        ax=None,
        window=None,
        strides=None,
        performance_metric="d_prime",
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
        peak_performance = self.performance_session_type(
            session_type,
            window=window,
            strides=strides,
            performance_metric=performance_metric,
            downsample_trials=downsample_trials,
        )
        ylabels = {
            "CRs": "Correct rejection rate",
            "hits": "Hit rate",
            "d_prime": "d'",
        }
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 4.75))
            label_y = True
        else:
            fig = ax.get_figure()
            label_y = False

        self.scatter_box(peak_performance, ax=ax)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(ages, rotation=45)
        if label_y:
            ax.set_ylabel(ylabels[performance_metric])
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        # ax = beautify_ax(ax)
        fig.tight_layout()

        return peak_performance, fig

    def peak_perf_to_df(self, peak_performance, label="CRs"):
        """
        Convert peak_performance from plot_best_performance() into
        a df.

        :param peak_performance:
        :param label:
        :return:
        """
        df = pd.DataFrame(
            np.hstack((peak_performance[age] for age in ages)),
            index=np.hstack([self.meta["grouped_mice"][age] for age in ages]),
            columns=[label],
        )

        df["aged"] = [self.meta["aged"][mouse] for mouse in df.index]

        return df

    def plot_peak_performance_all_sessions(
        self,
        window=None,
        strides=None,
        performance_metric="CRs",
        downsample_trials=False,
        sessions=None,
        plot_line=False,
        show_plot=True,
        ages_to_plot=None,
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

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in sessions
        ]

        ylabels = {
            "d_prime": "d'",
            "CRs": "Correct rejection rate",
            "hits": "Hit rate",
        }
        performance = {
            session_type: self.performance_session_type(
                session_type,
                window=window,
                strides=strides,
                performance_metric=performance_metric,
                downsample_trials=downsample_trials,
            )
            for session_type in sessions
        }
        if show_plot:
            if plot_line:
                fig, ax = plt.subplots(figsize=(4, 4))
                for age, color in zip(ages_to_plot, plot_colors):
                    data = np.hstack(
                        [
                            np.asarray(performance[session][age])[:, np.newaxis]
                            for session in sessions
                        ]
                    )

                    ax.plot(session_labels, data.T, color=color, alpha=0.5)
                    errorfill(
                        session_labels,
                        np.nanmean(data, axis=0),
                        yerr=sem(data, axis=0),
                        color=color,
                        ax=ax,
                    )
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]
                if performance_metric in ["hits", "CRs"]:
                    ax.set_ylim([0, 1])
                ax.set_ylabel(ylabels[performance_metric], fontsize=22)
                fig.tight_layout()
            else:
                fig, axs = plt.subplots(1, len(sessions), sharey=True)

                for ax, session, title in zip(axs, sessions, session_labels):
                    self.plot_performance_session_type(
                        session_type=session,
                        ax=ax,
                        window=window,
                        strides=strides,
                        performance_metric=performance_metric,
                        downsample_trials=downsample_trials,
                    )
                    ax.set_xticks([])
                    ax.set_title(title, fontsize=16)
                    [ax.spines[side].set_visible(False) for side in ["top", "right"]]
                axs[0].set_ylabel(ylabels[performance_metric])
                fig.tight_layout()
                fig.subplots_adjust(wspace=0)
                self.set_age_legend(fig)
        else:
            fig = None

        return performance, fig

    def performance_to_df(self, performance):
        """
        Convert output of plot_peak_performance_all_sessions() to df.

        :param performance:
        :return:
        """
        df = pd.DataFrame()
        for session in performance.keys():
            df[session] = pd.DataFrame(
                np.hstack((performance[session][age] for age in ages)),
                index=np.hstack([self.meta["grouped_mice"][age] for age in ages]),
            )

        df["aged"] = [self.meta["aged"][mouse] for mouse in df.index]

        mice_, ages_, metric_, session_ = [], [], [], []
        for session in performance.keys():

            for age in ages:
                mice_.extend(self.meta["grouped_mice"][age])
                ages_.extend([age for mouse in self.meta["grouped_mice"][age]])
                metric_.extend(performance[session][age])
                session_.extend([session for mouse in self.meta["grouped_mice"][age]])

        long_df = pd.DataFrame(
            {
                "mice": mice_,
                "age": ages_,
                "session": session_,
                "metric": metric_,
            }
        )

        return df, long_df

    def aged_performance_anova(
        self,
        performance_metric="d_prime",
        sessions=["Goals" + str(i) for i in np.arange(1, 5)],
        show_plot=True
    ):
        d_prime = self.plot_peak_performance_all_sessions(
            performance_metric=performance_metric, sessions=sessions, show_plot=show_plot
        )[0]

        df = self.performance_to_df(d_prime)[1]
        anova_df = pg.rm_anova(
            df, dv="metric", subject="mice", within=["session", "age"]
        )

        return anova_df

    def performance_anova(self, age, performance_metric="d_prime", sessions=None):
        if sessions is None:
            sessions = self.meta["session_types"]

        behavior, fig = self.plot_peak_performance_all_sessions(
            performance_metric=performance_metric,
            sessions=sessions,
            plot_line=True,
            ages_to_plot=age,
        )
        df = self.performance_to_df(behavior)[1]
        df = df.loc[df["age"] == age]

        anova_df = pg.rm_anova(
            df,
            dv="metric",
            subject="mice",
            within="session",
        )

        pairwise_df = df.pairwise_ttests(
            dv="metric", within="session", padjust="fdr_bh", subject="mice"
        )

        return df, anova_df, pairwise_df, fig

    def compare_CR_and_hit_diff(self, age="young", metrics=["hits", "CRs"]):
        if type(metrics) == str:
            metrics = [metrics]

        color = age_colors[ages.index(age)]
        diffs = dict()
        xticklabel = {"CRs": "Correct\nrejection rate", "hits": "Hit rate"}

        for metric in metrics:
            behavior = self.plot_peak_performance_all_sessions(
                performance_metric=metric,
                sessions=["Goals4", "Reversal"],
                show_plot=False,
            )[0]

            diffs[metric] = np.asarray(behavior["Reversal"][age]) - np.asarray(
                behavior["Goals4"][age]
            )

        diffs["mice"] = self.meta["grouped_mice"][age]
        diffs = pd.DataFrame(diffs)

        fig, axs = plt.subplots(
            1, len(metrics), figsize=(2 * len(metrics), 4.8), sharey=True
        )
        if len(metrics) == 1:
            axs = [axs]
        for ax, metric in zip(axs, metrics):
            ax.bar(
                0,
                diffs[metric].mean(),
                yerr=diffs[metric].sem(),
                zorder=1,
                color=color,
                ecolor="k",
                edgecolor="k",
                error_kw=dict(lw=1, capsize=5, capthick=1, zorder=0),
            )
            ax.scatter(
                jitter_x(np.zeros_like(diffs[metric])),
                diffs[metric],
                edgecolor="k",
                facecolor=color,
                zorder=2,
                s=100,
                alpha=0.5,
            )
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_xticks([0])
            ax.set_xticklabels([xticklabel[metric]], rotation=45)
        axs[0].set_ylabel("Performance change from\nTraining4 to Reversal")
        fig.tight_layout()

        return diffs, fig

    def scatter_box(self, data, ylabel="", ax=None, ages_to_plot=None, xticks=[]):
        """
        Make boxplot split into aged versus young. Scatter plot individual data points on top.

        :parameters
        ---
        data: dict
            Data in array-likes, split into "young" and "old".

        """
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        boxes = ax.boxplot(
            [data[age] for age in ages_to_plot],
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
                s=100,
            )
            for i, (age, color) in enumerate(zip(ages_to_plot, plot_colors))
        ]

        color_boxes(boxes, plot_colors)
        ax.set_xticklabels(xticks, rotation=45)
        ax.set_ylabel(ylabel)

        return fig

    def plot_perseverative_licking(
        self, show_plot=True, binarize=True, groupby="age", age_to_plot="young"
    ):
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

        errors = [perseverative_errors, unforgiveable_errors]
        if show_plot:
            ylabel = {True: "Proportion of trials", False: "Mean licks per trial"}
            if groupby == "error":
                fig, axs = plt.subplots(1, 2, sharey=True)
                fig.subplots_adjust(wspace=0)

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
            else:
                fig, ax = plt.subplots(figsize=(3, 4.8))
                color = self.ages_to_plot_parser(age_to_plot)[1][0]
                ax.bar(
                    [0, 1],
                    [np.nanmean(error[age_to_plot]) for error in errors],
                    yerr=[sem(error[age_to_plot]) for error in errors],
                    zorder=1,
                    color=color,
                    ecolor="k",
                    edgecolor="k",
                    error_kw=dict(lw=1, capsize=5, capthick=1, zorder=0),
                )

                ax.plot(
                    [
                        jitter_x(np.ones_like(error[age_to_plot]) * i)
                        for i, error in zip([0, 1], errors)
                    ],
                    [error[age_to_plot] for error in errors],
                    "o-",
                    color="k",
                    markerfacecolor=color,
                    zorder=2,
                    markersize=10,
                    alpha=0.5,
                )
                print(wilcoxon(*[error[age_to_plot] for error in errors]))

                ax.set_xticks([0, 1])
                ax.set_xticklabels(
                    ["Perseverative\nerrors", "Regular\nerrors"],
                    rotation=45,
                    fontsize=18,
                )
                ax.set_ylabel(ylabel[binarize], fontsize=22)
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]
                fig.tight_layout()

        return perseverative_errors, unforgiveable_errors, fig

    def plot_perseverative_licking_over_session(
        self,
        session_type="Reversal",
        window_size=6,
        trial_interval=2,
        show_plot=True,
        binarize_licks=True,
        ages_to_plot="young",
    ):
        """
        Plot perseverative and unforgiveable errors averaged across trial windows


        """
        n_trials = np.max(
            [
                self.data[mouse][session_type].behavior.data["ntrials"]
                for mouse in self.meta["mice"]
            ]
        )
        perseverative_errors = dict()
        regular_errors = dict()
        for age in ages:
            perseverative_errors[age] = [[] for mouse in self.meta["grouped_mice"][age]]
            regular_errors[age] = [[] for mouse in self.meta["grouped_mice"][age]]

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
                    regular_errors[age][i].append(
                        np.sum(licks_this_window[:, other_ports])
                        / (n_other * licks_this_window.shape[0])
                    )

        perseverative_errors = {
            age: stack_padding(perseverative_errors[age]) for age in ages
        }
        regular_errors = {age: stack_padding(regular_errors[age]) for age in ages}

        if show_plot:
            ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
                ages_to_plot
            )
            ylabel = "Error rate" if binarize_licks else "Average number of licks"
            if session_type == "Reversal":
                fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
                fig.subplots_adjust(wspace=0)

                for ax, rate, title in zip(
                    axs,
                    [perseverative_errors, regular_errors],
                    ["Perseverative\nerrors", "Regular\nerrors"],
                ):
                    se = {age: sem(rate[age], axis=0) for age in ages_to_plot}
                    m = {age: np.nanmean(rate[age], axis=0) for age in ages_to_plot}
                    for c, age in zip(plot_colors, ages_to_plot):
                        ax.plot(rate[age].T, color=c, alpha=0.2)
                        errorfill(
                            range(m[age].shape[0]), m[age], se[age], color=c, ax=ax
                        )
                        ax.set_title(title)
                    ax.set_xticks([int(x) for x in ax.get_xlim()])
                    ax.set_xticklabels([1, n_trials])
                    [ax.spines[side].set_visible(False) for side in ["top", "right"]]
                    fig.supxlabel("Trials")
                    fig.supylabel(ylabel)
            else:
                fig, ax = plt.subplots()
                se = {age: sem(regular_errors[age], axis=0) for age in ages_to_plot}
                m = {
                    age: np.nanmean(regular_errors[age], axis=0) for age in ages_to_plot
                }
                for c, age in zip(plot_colors, ages_to_plot):
                    ax.plot(regular_errors[age].T, color=c, alpha=0.5)
                    errorfill(range(m[age].shape[0]), m[age], se[age], color=c, ax=ax)
                ax.set_xlabel("Trial blocks")
                ax.set_ylabel(ylabel)

            fig.tight_layout()
        return perseverative_errors, regular_errors, fig

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

        fig, ax = plt.subplots(figsize=(3, 4.75))
        fig = self.scatter_box(trials, "Trials", ax=ax, xticks=ages)
        ax = fig.axes[0]
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        return trials, fig

    def compare_licks(self, session_type, exclude_rewarded=False):
        licks = {group: [] for group in ages}
        for group in ages:
            for mouse in self.meta["grouped_mice"][group]:
                if exclude_rewarded:
                    ports = ~self.data[mouse][session_type].behavior.data[
                        "rewarded_ports"
                    ]
                else:
                    ports = np.ones_like(
                        self.data[mouse][session_type].behavior.data["rewarded_ports"],
                        dtype=bool,
                    )
                licks[group].append(
                    np.sum(
                        self.data[mouse][session_type].behavior.data["all_licks"][
                            :, ports
                        ]
                    )
                )

        fig, ax = plt.subplots(figsize=(3, 4.75))
        fig = self.scatter_box(licks, "Licks", ax=ax, xticks=ages)
        ax = fig.axes[0]
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        return licks, fig

    ############################ ACTIVITY FUNCTIONS ##########################

    def percent_fading_neurons(
        self,
        mouse,
        session_type,
        x="trial",
        z_threshold=None,
        x_bin_size=1,
        alpha="sidak",
    ):
        trends, binned_activations, slopes, _ = self.find_activity_trends(
            mouse,
            session_type,
            x=x,
            z_threshold=z_threshold,
            x_bin_size=x_bin_size,
            alpha=alpha,
            data_type="S",
        )
        n_fading = len(trends["decreasing"])
        n_neurons = self.data[mouse][session_type].imaging["n_neurons"]
        p_fading = n_fading / n_neurons

        return p_fading

    def plot_percent_fading_neurons(
        self,
        session_pair=("Goals4", "Reversal"),
        x="trial",
        z_threshold=None,
        x_bin_size=1,
        alpha="sidak",
    ):
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
        data = {"age": [], "mouse": [], "p_fading": [], "session": []}
        for mouse in self.meta["mice"]:
            age = "aged" if mouse in aged_mice else "young"

            for session in session_pair:
                p_fading = self.percent_fading_neurons(
                    mouse,
                    session,
                    x=x,
                    z_threshold=z_threshold,
                    x_bin_size=x_bin_size,
                    alpha=alpha,
                )

                for key, value in zip(data.keys(), [age, mouse, p_fading, session]):
                    data[key].append(value)

        df = pd.DataFrame(data)

        fig, axs = plt.subplots(1, 2, sharey=True)
        fig.subplots_adjust(wspace=0)
        for age, color, ax in zip(ages, age_colors, axs):
            plot_me = [
                df["p_fading"].loc[
                    np.logical_and(df["session"] == session, df["age"] == age)
                ]
                for session in session_pair
            ]

            ax.plot(session_pair, plot_me, c=color)
            ax.set_title(age)
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_xlim([-0.5, 1.5])
        axs[0].set_ylabel("Percent fading neurons")
        fig.tight_layout()

        return df

    def hub_dropped_cells_metrics(
        self,
        mouse,
        graph_session,
        activity_rate_session=None,
        half=None,
        ax=None,
        graph_session_neuron_id=False,
        metric="event_rate",
    ):
        degree_df = self.find_degrees(mouse, graph_session)
        degree_df = self.categorize_hub_dropped_neurons(degree_df, time_bin=5)

        if activity_rate_session is None or activity_rate_session == graph_session:
            activity_rate_session = graph_session
        else:
            trimmed_map = self.get_cellreg_mappings(
                mouse,
                (graph_session, activity_rate_session),
                detected="either_day",
                neurons_from_session1=None,
            )[0]

        session = self.data[mouse][activity_rate_session]

        if metric == "event_rate":
            S_binary = session.imaging["S_binary"]
            if half is not None:
                S_binary = np.array_split(S_binary, 2, axis=1)[half]
            metrics = np.sum(S_binary, axis=1) / S_binary.shape[1]
        elif metric == "spatial_info":
            metrics = session.spatial.data["spatial_info_z"]

        categories = ["hub", "dropped"]
        mean_rates = {category: [] for category in categories}
        event_rates_df = pd.DataFrame()
        for ensemble_id in degree_df.ensemble_id.unique():
            in_ensemble = degree_df.ensemble_id == ensemble_id

            for q, category in zip([4, 1], categories):
                neurons_ = degree_df.loc[
                    np.logical_and(in_ensemble, degree_df.quartile == q), "neuron_id"
                ].unique()

                if activity_rate_session != graph_session:
                    col_num = 0 if graph_session_neuron_id else 1
                    reg_neurons = trimmed_map.loc[
                        trimmed_map.iloc[:, 0].isin(neurons_), trimmed_map.columns[1]
                    ].values
                    good_neurons = reg_neurons > -9999
                    neurons = reg_neurons[good_neurons]

                    neuron_labels = trimmed_map.loc[
                        trimmed_map.iloc[:, 0].isin(neurons_),
                        trimmed_map.columns[col_num],
                    ].values
                    neuron_labels = neuron_labels[good_neurons]
                else:
                    neurons = neurons_
                    neuron_labels = neurons_

                event_rates_df = pd.concat(
                    (
                        event_rates_df,
                        pd.DataFrame(
                            {
                                "mouse": mouse,
                                "session_id": activity_rate_session,
                                "ensemble_id": ensemble_id,
                                "neuron_id": neuron_labels,
                                "category": category,
                                f"{metric}": metrics[neurons],
                            }
                        ),
                    )
                )

                mean_rates[category].append(np.nanmean(metrics[neurons]))

        rates_df = pd.DataFrame(mean_rates).dropna()

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4.8))
            ax.set_ylabel("Transient rate")
        else:
            fig = ax.figure

        neuron_colors = cycle(["limegreen", "orange"])
        boxes = ax.boxplot(
            [rates_df[category] for category in categories],
            widths=0.75,
            patch_artist=True,
            zorder=0,
            showfliers=False,
        )
        x_ = []
        for i, category in zip([1, 2], categories):
            x = jitter_x(np.ones_like(rates_df[category]), 0.05) * i
            x_.append(x)
            ax.scatter(x, rates_df[category], color=next(neuron_colors), edgecolors="k")
        ax.plot(x_, [rates_df[category] for category in categories], color="k")
        color_boxes(boxes, ["limegreen", "orange"])

        ax.set_xticklabels(
            ["Hub\nneurons", "Dropped\nneurons"], rotation=45, fontsize=14
        )
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        return rates_df, event_rates_df

    def hub_dropped_cells_metrics_all_mice(
        self,
        mice_to_plot,
        graph_session="Reversal",
        activity_rate_session=None,
        half=None,
        axs=None,
        graph_session_neuron_id=True,
        metric="event_rate",
    ):
        if axs is None:
            fig, axs = plt.subplots(
                1, len(mice_to_plot), figsize=(len(mice_to_plot) * 2, 4.8), sharey=True
            )
        else:
            fig = axs[0].figure
        pvals = []
        ylabel = {
            "event_rate": "Ca2+ transient rate",
            "spatial_info": "Spatial info (z)",
        }
        df = pd.DataFrame()
        for mouse, ax in zip(mice_to_plot, axs):
            rates_df, long_df = self.hub_dropped_cells_metrics(
                mouse,
                graph_session,
                activity_rate_session,
                ax=ax,
                graph_session_neuron_id=graph_session_neuron_id,
                half=half,
                metric=metric,
            )
            pvals.append(ttest_rel(*[rates_df[col] for col in rates_df.columns]).pvalue)
            ax.set_title(f"Mouse {mouse[0]}")

            df = pd.concat((df, long_df))

        pvals = [np.round(x, 4) for x in multipletests(pvals, method="fdr_bh")[1]]
        print(f"{pvals}")

        df = df.reset_index(drop=True)

        axs[0].set_ylabel(ylabel[metric], fontsize=22)
        fig.tight_layout()

        return df, fig

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

    def plot_max_projs(self, mouse, session_types=None):
        if session_types is None:
            session_types = self.meta["session_types"]
        n_sessions = len(session_types)

        fig, axs = plt.subplots(1, n_sessions, figsize=(2 * n_sessions, 3))
        for ax, session_type in zip(axs, session_types):
            session_folder = self.data[mouse][session_type].meta["folder"]
            max_proj = open_minian(os.path.join(session_folder, "Miniscope"))[
                "max_proj"
            ]

            ax.imshow(max_proj, cmap="gray")
            ax.axis("off")
        fig.tight_layout()

        return fig

    ############################ PLACE FIELD FUNCTIONS ############################
    def percent_placecells(self):
        percent_pcs = pd.DataFrame()
        for mouse in self.meta["mice"]:
            for session in self.meta["session_types"]:
                pfs = self.data[mouse][session].spatial.data
                p = pd.DataFrame(
                    {
                        "age": "young"
                        if mouse not in self.meta["grouped_mice"]["aged"]
                        else "aged",
                        "mice": mouse,
                        "session_type": session,
                        "n_neurons": [pfs["n_neurons"]],
                        "n_pcs": [len(pfs["place_cells"])],
                        "p_pcs": [len(pfs["place_cells"]) / pfs["n_neurons"]],
                    }
                )
                percent_pcs = pd.concat((percent_pcs, p))

        return percent_pcs

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

    def PV_corr_pair(
        self, mouse, session_pair, nbins=125, velocity_threshold=7, corr="spearman"
    ):
        """
        Compute the PV correlation from a mouse for a session pair.

        """
        # Get placefields, make new ones if the specified parameters
        # don't match existing ones.
        pfs = {
            session: self.get_placefields(
                mouse, session, nbins=nbins, velocity_threshold=velocity_threshold
            )
            for session in session_pair
        }

        # Get correlation function.
        corr_fun = spearmanr if corr == "spearman" else pearsonr

        # Register neurons and place fields.
        trimmed_map = np.asarray(
            self.get_cellreg_mappings(mouse, session_pair, detected="everyday")[0]
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

    def compare_PV_corrs_by_bin(
        self,
        mouse,
        session_pair1,
        session_pair2,
        nbins=125,
        velocity_threshold=7,
        corr="spearman",
        colors=["darkgray", "steelblue"],
        show_plot=True,
    ):
        """
        Plot PV correlation values from one mouse from two session pairs. This allows comparison of
        PV correlations e.g. between Training3 x Training4 and Training4 x Reversal.

        """
        if nbins != 125:
            warnings.warn("Plotting reward site only works for nbins==125", UserWarning)

        session_pairs = [session_pair1, session_pair2]
        behavior_data = [
            self.data[mouse][session_pair[1]].behavior.data
            for session_pair in session_pairs
        ]

        port_bins = [
            find_reward_spatial_bins(
                data["df"]["lin_position"],
                np.asarray(data["lin_ports"]),
                spatial_bin_size_radians=0.05,
            )[0]
            for data in behavior_data
        ]

        rhos = [
            self.PV_corr_pair(
                mouse,
                session_pair,
                nbins=nbins,
                velocity_threshold=velocity_threshold,
                corr=corr,
            )[0]
            for session_pair in session_pairs
        ]

        if show_plot:
            fig, ax = plt.subplots()
            for rho, color, reward_locations, behavior in zip(
                rhos, colors, port_bins, behavior_data
            ):
                ax.plot(rho, color=color)

                [
                    ax.axvline(x=reward_location, color=color)
                    for reward_location in reward_locations[behavior["rewarded_ports"]]
                ]

                ax.set_xlabel("Spatial location")
                ax.set_ylabel("Correlation coefficient")

        return rhos, port_bins

    def compare_reward_PV_corrs(
        self,
        mouse,
        spatial_bin_window=5,
        nbins=125,
        velocity_threshold=0,
        corr="spearman",
        colors=["darkgray", "steelblue"],
    ):
        """
        Compare mean PV correlations of the bins around reward sites for two session pairs.

        :parameters
        ---
        mouse: str
            Mouse name.

        spatial_bin_window: int
            Number of spatial bins to take the mean across, flanking the reward site.

        """
        session_pairs = [("Goals3", "Goals4"), ("Goals4", "Reversal")]
        goals = {
            session: self.data[mouse][session].behavior.data["rewarded_ports"]
            for session in ["Goals4", "Reversal"]
        }

        binned_rhos, port_bins = self.compare_PV_corrs_by_bin(
            mouse,
            session_pairs[0],
            session_pairs[1],
            nbins=nbins,
            velocity_threshold=velocity_threshold,
            corr=corr,
            show_plot=False,
        )

        # Need to handle circularity of slices here.
        rhos = dict()
        never_rewarded = ~(goals["Goals4"] + goals["Reversal"])
        for rhos_pair, ports, session_pair in zip(
            binned_rhos, port_bins, session_pairs
        ):
            rhos[session_pair] = dict()
            current_goals = goals[session_pair[1]]

            rhos_at_rewards = [
                list(
                    islice(
                        cycle(rhos_pair),
                        reward - spatial_bin_window,
                        reward + spatial_bin_window,
                    )
                )
                for reward in ports[current_goals]
            ]

            rhos_at_nonrewards = [
                list(
                    islice(
                        cycle(rhos_pair),
                        reward - spatial_bin_window,
                        reward + spatial_bin_window,
                    )
                )
                for reward in ports[never_rewarded]
            ]

            rhos[session_pair]["currently_rewarded"] = np.nanmean(
                [np.nanmean(r) for r in rhos_at_rewards]
            )
            rhos[session_pair]["never_rewarded"] = np.nanmean(
                [np.nanmean(r) for r in rhos_at_nonrewards]
            )

            if session_pair[1] == "Reversal":
                rhos_at_previous = [
                    list(
                        islice(
                            cycle(rhos_pair),
                            reward - spatial_bin_window,
                            reward + spatial_bin_window,
                        )
                    )
                    for reward in ports[goals["Goals4"]]
                ]
                rhos[session_pair]["previously_rewarded"] = np.nanmean(
                    [np.nanmean(r) for r in rhos_at_previous]
                )

        return rhos

    def rate_remap_scores(
        self,
        mouse,
        session_type,
        place_cells_only=False,
        field_threshold=0.9,
        ports=None,
        spatial_bin_window=5,
    ):
        # Grab session data.
        session = self.data[mouse][session_type]
        spatial_data = session.spatial.data
        age = "aged" if mouse in self.meta["grouped_mice"]["aged"] else "young"
        nbins = spatial_data["placefields_normalized"].shape[1]

        if place_cells_only:
            neurons = session.spatial.data["place_cells"]
        else:
            neurons = np.arange(session.imaging["n_neurons"])

        # Define list of valid field bins.
        if ports is None:
            bin_range = np.arange(0, nbins)

        else:
            if ports == "previously rewarded":
                assert (
                    session_type == "Reversal"
                ), "Session type must be Reversal to look at previously rewarded bins"

                reward_session = self.data[mouse]["Goals4"].behavior.data

            elif ports == "newly rewarded":
                assert (
                    session_type == "Reversal"
                ), "Session type must be Reversal to look at newly rewarded bins"

                reward_session = self.data[mouse]["Reversal"].behavior.data

            elif ports == "original rewards":
                assert (
                    "Goals" in session_type
                ), "Session type must be a Goals session to look at original reward bins"

                reward_session = self.data[mouse]["Goals4"].behavior.data

            else:
                raise NotImplementedError

            reward_bins, bins = find_reward_spatial_bins(
                reward_session["df"]["lin_position"],
                np.asarray(reward_session["lin_ports"])[
                    reward_session["rewarded_ports"]
                ],
                spatial_bin_size_radians=0.05,
            )

            bin_range = []
            for bin in reward_bins:
                temp = [
                    i % len(bins)
                    for i in range(
                        bin - spatial_bin_window, bin + spatial_bin_window + 1
                    )
                ]

                bin_range.extend(temp)

        remap_scores = []
        for field, raster in zip(
            spatial_data["placefields_normalized"][neurons],
            spatial_data["rasters"][neurons],
        ):

            if field_threshold is not None:
                field_bins = define_field_bins(field, field_threshold=field_threshold)
            else:
                field_bins = np.arange(0, nbins)

            if ports is not None:
                field_bins = field_bins[[bin in bin_range for bin in field_bins]]

            split_raster = np.array_split(raster, 2, axis=0)

            rates = [np.nanmean(half[:, field_bins]) for half in split_raster]
            remap_scores.append(np.abs(np.diff(rates)[0]) / np.sum(rates))

        remap_score_df = pd.DataFrame(
            {
                "mice": mouse,
                "age": age,
                "session_type": session_type,
                "neuron_id": neurons,
                "remap_scores": remap_scores,
            }
        )

        return remap_score_df

    def compile_remap_scores(
        self,
        place_cells_only=True,
        session_types=["Goals4", "Reversal"],
        field_threshold=0.9,
        ports=None,
        spatial_bin_window=5,
    ):
        remap_score_df = pd.DataFrame()
        if ports is None:
            ports = [None, None]

        for mouse in self.meta["mice"]:
            for session_type, p in zip(session_types, ports):
                remap_scores = self.rate_remap_scores(
                    mouse,
                    session_type,
                    place_cells_only=place_cells_only,
                    field_threshold=field_threshold,
                    ports=p,
                    spatial_bin_window=spatial_bin_window,
                )
                remap_score_df = pd.concat((remap_score_df, remap_scores))

        return remap_score_df.dropna()

    def plot_remap_score_by_mouse(self, **kwargs):
        remap_score_df = self.compile_remap_scores(*kwargs)

        fig, axs = plt.subplots(1, 2, sharey=True)
        fig.subplots_adjust(wspace=0.1)
        for age, ax, color in zip(ages, axs, age_colors):
            mice = self.meta["grouped_mice"][age]
            positions = [
                np.arange(start, start + 3 * len(mice), 3) for start in [-0.5, 0.5]
            ]
            label_positions = np.arange(0, 3 * len(mice), 3)

            for session_type, position in zip(session_types, positions):
                y = [
                    remap_score_df.loc[
                        np.logical_and(
                            remap_score_df["mice"] == mouse,
                            remap_score_df["session_type"] == session_type,
                        ),
                        "remap_scores",
                    ]
                    for mouse in mice
                ]
                boxes = ax.boxplot(y, positions=position, patch_artist=True)
                color_boxes(boxes, color)

            ax.set_xticks(label_positions)
            ax.set_xticklabels(mice, rotation=45)
        axs[0].set_ylabel("Remap scores")
        plt.setp(axs[1].get_yticklabels(), visible=False)

        return remap_score_df

    def plot_remap_score_means(
        self, ages_to_plot=None, session_types=["Goals4", "Reversal"], **kwargs
    ):
        remap_score_df = self.compile_remap_scores(
            session_types=session_types, **kwargs
        )

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        fig, axs = plt.subplots(
            1, n_ages_to_plot, sharey=True, figsize=(3 * n_ages_to_plot, 4.8)
        )

        mean_df = remap_score_df.groupby(["mice", "session_type"]).mean()[
            "remap_scores"
        ]
        if n_ages_to_plot == 1:
            axs = [axs]

        for ax, age, color in zip(axs, ages_to_plot, plot_colors):
            mice = self.meta["grouped_mice"][age]

            y = [mean_df.loc[mice, session_type] for session_type in session_types]
            print(f"{age}: ")
            print(wilcoxon(*y))
            print("")
            ax.bar(
                [0, 1],
                [np.nanmean(x) for x in y],
                yerr=[sem(x) for x in y],
                zorder=1,
                color=color,
                ecolor="k",
                edgecolor="k",
                error_kw=dict(lw=1, capsize=5, capthick=1, zorder=0),
            )

            y = np.vstack(y).T
            for mouse_data in y:
                ax.plot(
                    jitter_x([0, 1], 0.05),
                    mouse_data,
                    "o-",
                    color="k",
                    markerfacecolor=color,
                    zorder=2,
                    markersize=10,
                    alpha=0.4,
                )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [
                    session_type.replace("Goals", "Training")
                    for session_type in session_types
                ],
                rotation=45,
            )
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_ylim([0, 0.8])
        axs[0].set_ylabel("Rate remap scores", fontsize=22)
        fig.tight_layout()

        return remap_score_df, fig

    def test_rate_remap_sig(self, remap_score_df, session_types=["Goals4", "Reversal"]):
        mean_df = remap_score_df.groupby(["mice", "session_type"]).mean()[
            "remap_scores"
        ]

        for age in ages:
            mice = self.meta["grouped_mice"][age]

            data = [mean_df.loc[mice, session_type] for session_type in session_types]

            stat, pval = wilcoxon(data[0], data[1])

            print(
                f"{age} mice: {session_types[0]} vs {session_types[1]}, "
                f"W={stat} p={np.round(pval, 3)}"
            )

    def plot_reward_PV_corrs_v1(self, mice):
        """
        Plot PV correlations around reward locations for each mouse, split into session pairs for each subplot,
        then into currently/never rewarded within each subplot.

        :param mice:
        :return:
        """
        session_pairs = [("Goals3", "Goals4"), ("Goals4", "Reversal")]
        rhos = {mouse: self.compare_reward_PV_corrs(mouse) for mouse in mice}
        xlabels = [
            f'{session_pair[0].replace("Goals", "Training")} x {session_pair[1].replace("Goals", "Training")}'
            for session_pair in session_pairs
        ]

        fig, axs = plt.subplots(1, 2, sharey=True)
        fig.subplots_adjust(wspace=0)

        for session_pair, xlabel, ax in zip(session_pairs, xlabels, axs):
            reward_types = rhos[mice[0]][session_pair].keys()
            xticks = [
                str(reward_str).replace("_", " \n") for reward_str in reward_types
            ]

            data = []
            for reward_type in reward_types:
                data.append([rhos[mouse][session_pair][reward_type] for mouse in mice])

            ax.boxplot(data)
            ax.set_xticklabels(xticks, rotation=45)
            ax.set_xlabel(xlabel)

        axs[0].set_ylabel("PV correlation coefficient")
        fig.tight_layout()

        return rhos

    def plot_reward_PV_corrs_v2(
        self,
        age_to_plot,
        locations_to_plot=[
            "currently_rewarded",
            "never_rewarded",
            "previously_rewarded",
        ],
    ):
        """
        Plot PV correlations around reward locations for each mouse, split into currently/never/previously
        rewarded for each subplot, then into session pair within each subplot.

        :param mice:
        :return:
        """
        mice = self.meta["grouped_mice"][age_to_plot]
        color = self.ages_to_plot_parser(age_to_plot)[1][0]
        session_pairs = [("Goals3", "Goals4"), ("Goals4", "Reversal")]
        rhos = {mouse: self.compare_reward_PV_corrs(mouse) for mouse in mice}
        xticks = [
            f'{session_pair[0].replace("Goals", "Training")} x \n{session_pair[1].replace("Goals", "Training")}'
            for session_pair in session_pairs
        ]

        n_plots = len(locations_to_plot)
        fig, axs = plt.subplots(1, n_plots, sharey=True, figsize=(n_plots * 3, 4.8))
        if n_plots == 1:
            axs = [axs]
        fig.subplots_adjust(wspace=0)

        for ax, reward_type in zip(axs, locations_to_plot):
            data = []
            if reward_type != "previously_rewarded":
                for session_pair in session_pairs:
                    data.append(
                        [rhos[mouse][session_pair][reward_type] for mouse in mice]
                    )

            else:
                for session_pair, reward_type_ in zip(
                    session_pairs, ["currently_rewarded", reward_type]
                ):
                    data.append(
                        [rhos[mouse][session_pair][reward_type_] for mouse in mice]
                    )
            ax.bar(
                [0, 1],
                [np.nanmean(x) for x in data],
                yerr=[sem(x) for x in data],
                zorder=1,
                color=color,
                ecolor="k",
                edgecolor="k",
                error_kw=dict(lw=1, capsize=5, capthick=1, zorder=0),
            )
            print(wilcoxon(data[0], data[1]))

            ax.plot(
                [jitter_x(np.ones_like(x) * i) for i, x in zip([0, 1], data)],
                data,
                "o-",
                color="k",
                markerfacecolor=color,
                zorder=2,
                markersize=10,
                alpha=0.4,
            )

            ax.set_ylim([0, 0.6])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(xticks, rotation=45, fontsize=14)
            if n_plots > 1:
                ax.set_xlabel(reward_type.replace("_", " \n"))
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]

        axs[0].set_ylabel("Spatial PV\ncorrelation coefficient", fontsize=22)
        [axs[-1].spines[side].set_visible(False) for side in ["top", "right"]]

        fig.tight_layout()

        return rhos, fig

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
            ax.set_xticks(range(len(self.meta["session_types"])))
            ax.set_yticks(range(len(self.meta["session_types"])))
            ax.set_xticklabels(self.meta["session_labels"], rotation=45, fontsize=22)
            ax.set_yticklabels(self.meta["session_labels"], fontsize=22)
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
        cbar.set_label("Spatial PV correlation (Spearman rho)")

        return fig

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
        ages_to_plot=None,
    ):
        """
        For the specified sessions, plot the PV correlation coefficient between those sessions,
        separated by age.

        """
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        data = {session_pair: dict() for session_pair in session_pairs}
        for session_pair in session_pairs:
            s1 = self.meta["session_types"].index(session_pair[0])
            s2 = self.meta["session_types"].index(session_pair[1])
            data[session_pair] = dict()
            for age in ages:
                data[session_pair][age] = []
                for mouse in self.meta["grouped_mice"][age]:
                    data[session_pair][age].append(corr_matrices[mouse][s1, s2])

        fig, axs = plt.subplots(
            1, n_ages_to_plot, sharey=True, figsize=(3 * n_ages_to_plot, 4.8)
        )
        if n_ages_to_plot == 1:
            axs = [axs]

        for ax, age, color in zip(axs, ages_to_plot, plot_colors):
            ax.bar(
                [0, 1],
                [np.mean(data[session_pair][age]) for session_pair in session_pairs],
                yerr=[sem(data[session_pair][age]) for session_pair in session_pairs],
                zorder=1,
                color=color,
                ecolor="k",
                edgecolor="k",
                error_kw=dict(lw=1, capsize=5, capthick=1, zorder=0),
            )

            y = np.vstack([data[session_pair][age] for session_pair in session_pairs]).T
            for y_ in y:
                ax.plot(
                    jitter_x([0, 1], 0.05),
                    y_,
                    "o-",
                    color="k",
                    markerfacecolor=color,
                    zorder=1,
                    markersize=10,
                    alpha=0.5,
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [
                    f'{session_pair[0].replace("Goals", "Training")} x \n'
                    f'{session_pair[1].replace("Goals", "Training")}'
                    for session_pair in session_pairs
                ],
                rotation=45,
                fontsize=14,
            )
            ax.set_ylim([0, 0.6])

            if n_ages_to_plot == 2:
                ax.set_title(age)

            [ax.spines[side].set_visible(False) for side in ["top", "right"]]

        axs[0].set_ylabel("Spatial PV\ncorrelation coefficient", fontsize=22)
        fig.tight_layout()

        df = pd.concat(
            [
                pd.concat({k: pd.Series(v) for k, v in data[pair].items()})
                for pair in session_pairs
            ],
            axis=1,
            keys=session_pairs,
        )

        return data, df, fig

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
        performance = self.aggregate_behavior_over_trials(
            window=None,
            strides=None,
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
                color_boxes(boxes, color)

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
                raise NotImplementedError("mode must be holoviews, png, or scroll")
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
            ScrollObj = ScrollPlot(
                plot_daily_rasters,
                current_position=0,
                nrows=len(session_types),
                ncols=2,
                rasters=daily_rasters,
                tuning_curves=placefields,
                titles=ax_titles,
                figsize=(5, 9),
            )

            fig = ScrollObj.fig
        else:
            fig = None

        # List of dictionaries. Do HoloMap(daily_rasters) in a
        # jupyter notebook.
        return daily_rasters, fig

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
            ylabel = "Spatial info [z]"
        else:
            ylabel = ""

        if aggregate_mode not in ["median", "mean", "all"]:
            raise NotImplementedError("Invalid aggregate_mode")

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

    def plot_all_spatial_info(self, aggregate_mode="mean", place_cells_only=False):
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

    def spatial_info_anova(self, aggregate_mode="mean", place_cells_only=False):
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

    def plot_reliabilities(
        self,
        session_type,
        field_threshold=0.5,
        show_plot=True,
        data_type="ensembles",
        bin_size=0.05,
    ):
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

                if "cells" in data_type:
                    spatial_data = session.spatial.data

                    if data_type == "cells":
                        units = range(session.imaging["n_neurons"])
                    elif data_type == "place_cells":
                        units = session.spatial.data["place_cells"]

                elif "ensembles" in data_type:
                    # Compute rasters if not already done.
                    if (
                        "rasters" not in session.assemblies["fields"].data
                        or session.assemblies["fields"].meta["raster_bin_size"]
                        != bin_size
                    ):
                        self.make_ensemble_raster(
                            mouse, session_type, bin_size=bin_size, running_only=False
                        )
                    spatial_data = session.assemblies["fields"].data
                    if data_type == "ensembles":
                        units = range(session.assemblies["significance"].nassemblies)
                    elif data_type == "spatial_ensembles":
                        units = np.where(
                            self.get_spatial_ensembles(mouse, session_type)
                        )[0]

                else:
                    raise NotImplementedError

                reliabilities_ = [
                    session.placefield_reliability(
                        spatial_data,
                        i,
                        field_threshold=field_threshold,
                        even_split=True,
                        split=1,
                        show_plot=False,
                    )
                    for i in units
                ]
                reliabilities[age].append(reliabilities_)

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, color, ax in zip(ages, age_colors, axs):
                mice = self.meta["grouped_mice"][age]
                boxes = ax.boxplot(reliabilities[age], patch_artist=True)
                color_boxes(boxes, color)

                if age == "aged":
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("Place cell reliability")
                ax.set_xticklabels(mice, rotation=45)

        return reliabilities

    def reliabilities_df(self, session_type, **kwargs):
        reliabilities = self.plot_reliabilities(session_type, show_plot=False, **kwargs)
        reliabilities_df = pd.DataFrame()
        for age in ages:
            for mouse, rel_list in zip(
                self.meta["grouped_mice"][age], reliabilities[age]
            ):
                mouse_dict = {
                    "age": age,
                    "mouse": mouse,
                    "units": np.arange(len(rel_list)),
                    "reliabilities": rel_list,
                    "session": session_type,
                }

                reliabilities_df = pd.concat(
                    (reliabilities_df, pd.DataFrame(mouse_dict))
                )

        return reliabilities_df

    def fading_ensemble_reliabilities_on_other_days(
        self, other_session="Goals4", origin_session="Reversal", ages_to_plot="young"
    ):
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        reliabilities_df = pd.DataFrame()
        session_pair = [origin_session, other_session]
        for session in session_pair:
            reliabilities_df = pd.concat(
                (self.reliabilities_df(session), reliabilities_df)
            )

        trends = dict()
        registered_ensembles = dict()
        for age in ages_to_plot:
            for mouse in self.meta["grouped_mice"][age]:
                trends[mouse] = self.find_activity_trends(mouse, origin_session)[0]
                registered_ensembles[mouse] = self.match_ensembles(mouse, session_pair)

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6.4, 7))
        xticklabels = {
            other_session: ["Fading\n(during Reversal)", "No trend\n(during Reversal)"],
            origin_session: ["Fading", "No trend"],
        }
        for ax, session in zip(axs, [other_session, origin_session]):
            for age in ages_to_plot:
                colors = cm.rainbow(
                    np.linspace(0, 1, len(self.meta["grouped_mice"][age]))
                )
                edgecolor = "k" if age == "aged" else "face"
                for mouse, c in zip(self.meta["grouped_mice"][age], colors):
                    if session == other_session:
                        poor_matches = np.where(
                            registered_ensembles[mouse]["poor_matches"]
                        )[0]
                    else:
                        poor_matches = []
                    fading = [
                        f for f in trends[mouse]["decreasing"] if f not in poor_matches
                    ]
                    no_trend = [
                        n for n in trends[mouse]["no trend"] if n not in poor_matches
                    ]

                    if session == other_session:
                        fading = registered_ensembles[mouse]["matches"][fading]
                        no_trend = registered_ensembles[mouse]["matches"][no_trend]

                    # Get session data.
                    mouse_idx = reliabilities_df["mouse"] == mouse
                    session_idx = reliabilities_df["session"] == session
                    mouse_data = reliabilities_df.loc[
                        np.logical_and(mouse_idx, session_idx)
                    ]

                    # Index fading / no trend ensembles.
                    fading_rels = mouse_data.loc[
                        mouse_data["units"].isin(fading), "reliabilities"
                    ]
                    no_trend_rels = mouse_data.loc[
                        mouse_data["units"].isin(no_trend), "reliabilities"
                    ]

                    # Plot.
                    x = np.hstack(
                        (
                            jitter_x(np.zeros_like(fading_rels), 0.05),
                            jitter_x(np.ones_like(no_trend_rels), 0.05),
                        )
                    )
                    ax.scatter(
                        x,
                        np.hstack((fading_rels, no_trend_rels)),
                        c=c,
                        edgecolors=edgecolor,
                        alpha=0.2,
                    )
                    ax.plot(
                        [0, 1],
                        [np.nanmean(fading_rels), np.nanmean(no_trend_rels)],
                        "o-",
                        c=c,
                        markeredgecolor=c,
                        markersize=10,
                    )

                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(xticklabels[session], rotation=45)
                    [ax.spines[side].set_visible(False) for side in ["top", "right"]]
                    ax.set_title(session.replace("Goals", "Training"))

            axs[0].set_ylabel("Ensemble field reliabilities")
            fig.tight_layout()

        return reliabilities_df

    def scrollplot_fading_ensembles(self, mouse, session_type="Goals4"):
        trends = self.find_activity_trends(mouse, "Reversal")[0]
        registered_ensembles = self.match_ensembles(mouse, ("Reversal", session_type))
        poor_matches = np.where(registered_ensembles["poor_matches"])[0]
        fading = [f for f in trends["decreasing"] if f not in poor_matches]
        ensemble_numbers = registered_ensembles["matches"][fading]

        self.scrollplot_ensemble_rasters(mouse, session_type, subset=ensemble_numbers)

    def plot_reliabilities_comparisons(
        self,
        session_types=("Goals4", "Reversal"),
        field_threshold=0.5,
        show_plot=True,
        data_type="ensembles",
        ages_to_plot=None,
        bin_size=0.05,
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
                session_type,
                field_threshold=field_threshold,
                show_plot=False,
                data_type=data_type,
                bin_size=bin_size,
            )
            for age in ages:
                mean_reliabilities[session_type][age] = [
                    np.nanmean(r) for r in reliabilities[session_type][age]
                ]

        if show_plot:
            ylabel_data_type = {
                "ensembles": "Ensemble",
                "spatial_ensembles": "Spatial ensemble",
                "place_cells": "Place cell",
                "cells": "All cells",
            }
            fig, axs = plt.subplots(1, 2, sharey=True)
            fig.subplots_adjust(wspace=0)

            # Plot distributions for each mouse.
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
                    color_boxes(boxes, color)

                ax.set_xticks(label_positions)
                ax.set_xticklabels(mice, rotation=45)
            axs[0].set_ylabel(
                f"{ylabel_data_type[data_type]} spatial stability", fontsize=22
            )

            # Plot distributions of mean reliabilities.
            fig, axs = plt.subplots(1, len(session_types), sharey=True)
            fig.subplots_adjust(wspace=0)

            for session_type, ax in zip(session_types, axs):
                self.scatter_box(mean_reliabilities[session_type], ax=ax)
                ax.set_title(session_type)
            axs[0].set_ylabel(f"{ylabel_data_type[data_type]} spatial stability")

            # Group by age.
            if len(session_types) == 2:
                ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
                    ages_to_plot
                )
                fig, axs = plt.subplots(
                    1, n_ages_to_plot, sharey=True, figsize=(3.5 * n_ages_to_plot, 6)
                )

                if n_ages_to_plot == 1:
                    axs = [axs]
                for age, color, ax in zip(ages_to_plot, plot_colors, axs):
                    mean_reliabilities_by_session = [
                        mean_reliabilities[session_type][age]
                        for session_type in session_types
                    ]

                    boxes = ax.boxplot(
                        mean_reliabilities_by_session,
                        patch_artist=True,
                        widths=0.75,
                        zorder=0,
                        showfliers=False,
                    )
                    color_boxes(boxes, color)

                    data_points = np.vstack(mean_reliabilities_by_session).T
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
                    ax.set_xticklabels(
                        [
                            session_type.replace("Goals", "Training")
                            for session_type in session_types
                        ],
                        rotation=45,
                        fontsize=16,
                    )
                    [ax.spines[side].set_visible(False) for side in ["top", "right"]]

                    if n_ages_to_plot > 1:
                        ax.set_title(age)

                axs[0].set_ylabel(
                    f"{ylabel_data_type[data_type]}\nspatial stability", fontsize=22
                )
                fig.tight_layout()
                fig.subplots_adjust(wspace=0)
        else:
            fig = None

        return reliabilities, mean_reliabilities, fig

    def correlate_field_reliability_to_performance(
        self,
        performance_metric="d_prime",
        session_types=("Goals4", "Reversal"),
        field_threshold=0.9,
        window=5,
        data_type="ensembles",
        ages_to_plot=None,
        show_plot=True,
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
        reliabilities, mean_reliabilities, _ = self.plot_reliabilities_comparisons(
            session_types=session_types,
            field_threshold=field_threshold,
            show_plot=False,
            data_type=data_type,
        )

        performance = self.plot_performance_session_type(
            "Reversal",
            window=window,
            performance_metric=performance_metric,
            show_plot=False,
        )[0]

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        ylabel_data_type = {
            "ensembles": "Ensemble",
            "spatial_ensembles": "Spatial ensemble",
            "place_cells": "Place cell",
            "cells": "All cells",
        }

        if show_plot:
            fig, ax = plt.subplots()
            for age, color in zip(ages_to_plot, plot_colors):
                x = mean_reliabilities["Reversal"][age]
                y = performance[age]
                ax.scatter(x, y, color=color, edgecolors="k", s=100)

                # Plot fit line.
                z = np.polyfit(x, y, 1)
                y_hat = np.poly1d(z)(x)
                ax.plot(x, y_hat, color=color)

                [ax.spines[side].set_visible(False) for side in ["top", "right"]]

            ylabel = {
                "CRs": "correct rejection rate",
                "hits": "hit rate",
                "d_prime": "d'",
            }
            ax.set_ylabel(f"Peak {ylabel[performance_metric]}", fontsize=22)
            ax.set_xlabel(f"{ylabel_data_type[data_type]} stability", fontsize=22)
            fig.tight_layout()
        else:
            fig = None

        r, pvalue = dict(), dict()
        for age in ages:
            r[age], pvalue[age] = spearmanr(
                mean_reliabilities["Reversal"][age], performance[age]
            )

        r["combined"], pvalue["combined"] = spearmanr(
            np.hstack([mean_reliabilities["Reversal"][age] for age in ages]),
            np.hstack([performance[age] for age in ages]),
        )

        return r, pvalue, fig

    def snakeplot_placefields(
        self,
        mouse,
        session_type,
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
        spatial_bin_size_radians = self.data[mouse][session_type].spatial.meta[
            "bin_size"
        ]
        behavior_data = self.data[mouse][session_type].behavior

        session = self.data[mouse][session_type].spatial.data
        port_locations = np.asarray(behavior_data.data["lin_ports"])[
            behavior_data.data["rewarded_ports"]
        ]
        if neurons is None:
            neurons = np.arange(0, session["n_neurons"])
        if order is None:
            order = np.argsort(session["placefield_centers"][neurons])

        placefields = session["placefields_normalized"][neurons][order]

        if show_plot:
            if normalize:
                placefields = placefields / np.max(placefields, axis=1, keepdims=True)
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5.5))
            cmap = matplotlib.cm.get_cmap()
            cmap.set_bad(color="k")
            ax.imshow(placefields, cmap=cmap)

            if show_reward_sites:
                # Get reward locations
                reward_location_bins, bins = find_reward_spatial_bins(
                    session["x"],
                    port_locations,
                    spatial_bin_size_radians=spatial_bin_size_radians,
                )

                [
                    ax.axvline(x=reward_location_bin, color="w")
                    for reward_location_bin in reward_location_bins
                ]

                if session_type == "Reversal":
                    training_session = self.data[mouse]["Goals4"]
                    ex_reward_bins = find_reward_spatial_bins(
                        training_session.spatial.data["x"],
                        np.asarray(training_session.behavior.data["lin_ports"])[
                            training_session.behavior.data["rewarded_ports"]
                        ],
                        spatial_bin_size_radians=spatial_bin_size_radians,
                    )[0]

                    [
                        ax.axvline(x=reward_location_bin, color="w", alpha=0.3)
                        for reward_location_bin in ex_reward_bins
                    ]

            ax.axis("tight")
            ax.set_ylabel("Neuron #", fontsize=22)
            ax.set_xlabel("Location", fontsize=22)

            ax.set_xticks(ax.get_xlim())
            ax.set_xticklabels([0, 220])

        else:
            ax = None

        return ax, placefields, order

    def snakeplot_matched_placefields(
        self,
        mouse,
        session_types,
        sort_by_session=None,
        place_cells_only=False,
        place_cell_session=None,
    ):
        """
        Snake plot place fields that have been matched across sessions.

        """
        trimmed_map = self.get_cellreg_mappings(
            mouse, session_types, detected="everyday", neurons_from_session1=None
        )[0]
        trimmed_map = np.asarray(trimmed_map)

        if place_cell_session is None:
            place_cell_session = sort_by_session
        if place_cells_only:
            place_cells = self.data[mouse][
                session_types[place_cell_session]
            ].spatial.data["place_cells"]
            trimmed_map = trimmed_map[
                np.in1d(trimmed_map[:, place_cell_session], place_cells), :
            ]
        age = "aged" if mouse in self.meta["grouped_mice"]["aged"] else "young"

        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in session_types
        ]
        fig, axs = plt.subplots(
            1, len(session_types), figsize=(2 * len(session_types), 6)
        )
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
            ax.set_title(label, fontsize=22)
            ax.set_xlabel("")

            if i > 0:
                ax.set_ylabel("")
                ax.set_yticks([])
        fig.suptitle(f"Mouse {mouse[0]}")
        fig.supxlabel("Linearized position (cm)", fontsize=22)
        fig.tight_layout()

        return fig

    def reward_centered_field_density(
        self,
        mouse,
        session_type,
        place_cells_only=False,
        plot_window=(-5, 5),
        show_plot=True,
    ):
        session = self.data[mouse][session_type]

        # Filter cells.
        if place_cells_only:
            neurons = session.spatial.data["place_cells"]
        else:
            neurons = np.arange(session.imaging["n_neurons"])

        field_centers = session.spatial.data["placefield_centers"][neurons]
        # Find reward location.
        reward_bins, bins = find_reward_spatial_bins(
            session.behavior.data["df"]["lin_position"],
            np.asarray(session.behavior.data["lin_ports"])[
                session.behavior.data["rewarded_ports"]
            ],
            spatial_bin_size_radians=0.05,
        )

        bins_to_plot = []
        for reward_bin in reward_bins:
            normalized_bins = field_centers.copy()
            normalized_bins -= reward_bin
            normalized_bins = np.asarray(
                [math.remainder(x, 125) for x in normalized_bins]
            )  # for circularity

            bins_to_plot.extend(normalized_bins)

        bins_to_plot = np.asarray(bins_to_plot)
        bins_to_plot = bins_to_plot[np.isfinite(bins_to_plot)]

        if show_plot:
            fig, ax = plt.subplots()
            ax.hist(bins_to_plot, bins=125, density=True, histtype="step")
            ax.xlim(plot_window)
            ax.set_xlabel("Distance from reward site")
            ax.set_ylabel("Field density")
            ax.set_title(mouse)
            fig.tight_layout()

            print(
                chisquare(
                    np.histogram(bins_to_plot, range=plot_window, density=True)[0]
                )
            )

        return field_centers, bins_to_plot

    def plot_all_reward_field_densities(
        self,
        session_types,
        place_cells_only=False,
        plot_window=(-5, 5),
        ages_to_plot=None,
    ):
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        n_sessions = len(session_types)
        fig, axs = plt.subplots(
            n_ages_to_plot,
            n_sessions,
            sharey=True,
            sharex=True,
            figsize=(4 * n_sessions, 4.8 * n_ages_to_plot),
            squeeze=False,
        )

        for age, color, row_ax in zip(ages_to_plot, plot_colors, axs):
            hist_opts = {
                "density": True,
                "histtype": "step",
                "bins": np.arange(-62, 62),
                "edgecolor": color,
            }
            for ax, session_type in zip(row_ax, session_types):
                all_bins = []
                for mouse in self.meta["grouped_mice"][age]:
                    b = self.reward_centered_field_density(
                        mouse,
                        session_type,
                        place_cells_only=place_cells_only,
                        plot_window=plot_window,
                        show_plot=False,
                    )[1]
                    ax.hist(
                        b,
                        alpha=0.5,
                        **hist_opts,
                    )

                    all_bins.extend(b)

                all_bins = np.asarray(all_bins)
                ax.hist(
                    all_bins,
                    linewidth=2,
                    **hist_opts,
                )
                ax.set_xlim(plot_window)
                ax.set_xticks([plot_window[0], 0, plot_window[-1]])
                ax.set_xticklabels([x * 2 for x in ax.get_xticks()])
                ax.set_title(session_type.replace("Goals", "Training"))

                [ax.spines[side].set_visible(False) for side in ["top", "right"]]

                n, b = np.histogram(
                    all_bins,
                    bins=hist_opts["bins"],
                )[:2]
                distribution = [n[list(b).index(i)] for i in np.arange(*plot_window)]
                print(chisquare(distribution))
        fig.supxlabel("Distance from reward (cm)")
        fig.supylabel("Prop. cells with fields\nrel. to reward")
        fig.tight_layout()

        return fig

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
            raise NotImplementedError
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
        session_pairs=(("Goals4", "Goals3"), ("Goals4", "Reversal")),
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
        decoded_sessions = [s[1] for s in session_pairs]
        for key, session_pair in zip(decoded_sessions, session_pairs):

            decoding_errors[key] = self.find_decoding_error(
                mouse,
                session_pair,
                classifier=classifier,
                n_spatial_bins=n_spatial_bins,
                error_time_bin_size=error_time_bin_size,
                show_plot=False,
                predictors=predictors,
            )[0]
        time_bins = np.linspace(0, 1, len(decoding_errors[decoded_sessions[0]]))

        if axs is None:
            fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
        for errors, ax in zip(
            [decoding_errors[decoded_session] for decoded_session in decoded_sessions],
            axs,
        ):
            means = [np.nanmean(error) for error in errors]
            # sems = [np.nanstd(error) / np.sqrt(error.shape[0]) for error in errors]
            ax.plot(time_bins, means, color=color, alpha=0.2)

        return decoding_errors

    def within_session_spatial_decoder(
        self,
        mouse,
        session_type,
        classifier=GaussianNB,
        n_spatial_bins=125,
        predictors="ensembles",
        kfolds=5,
        n_shuffles=100,
        **classifier_kwargs,
    ):
        session = self.data[mouse][session_type]
        running = session.spatial.data["running"]
        fps = 15

        classifier = classifier(**classifier_kwargs)

        if predictors == "cells":
            neural_data = session.imaging["S_binary"]
        elif predictors == "ensembles":
            neural_data = zscore(session.assemblies["activations"], axis=0)
        else:
            raise NotImplementedError

        X = neural_data[:, running].T
        y = format_spatial_location_for_decoder(
            session.behavior.data["df"]["lin_position"].values[running],
            n_spatial_bins=n_spatial_bins,
            time_bin_size=1 / fps,
            fps=fps,
            classifier=classifier,
        )

        error = self.test_classifier(
            X,
            y,
            classifier=classifier,
            kfolds=kfolds,
            n_spatial_bins=n_spatial_bins,
            shuffle=False,
        )

        iterations = tqdm([i for i in np.arange(n_shuffles)])

        errors_shuffled = [
            self.test_classifier(
                X,
                y,
                kfolds=kfolds,
                classifier=classifier,
                n_spatial_bins=n_spatial_bins,
                shuffle=True,
            )
            for i in iterations
        ]

        return error, errors_shuffled

    def plot_spatial_decoder_errors_over_days(
        self, session_types=None, ages_to_plot="young", **kwargs
    ):
        if session_types is None:
            session_types = self.meta["session_types"]

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        errors = dict()
        errors_shuffled = dict()
        for session_type in session_types:
            errors[session_type] = dict()
            errors_shuffled[session_type] = dict()
            for age in ages_to_plot:
                errors[session_type][age] = []
                errors_shuffled[session_type][age] = []
                for mouse in self.meta["grouped_mice"][age]:
                    e, es = self.within_session_spatial_decoder(
                        mouse, session_type, **kwargs
                    )
                    errors[session_type][age].append(e)
                    errors_shuffled[session_type][age].append(es)

        xlabels = [session.replace("Goals", "Training") for session in session_types]
        fig, axs = plt.subplots(1, n_ages_to_plot, sharey=True)
        axs = [axs] if n_ages_to_plot == 1 else axs
        for ax, age, color in zip(axs, ages_to_plot, plot_colors):
            errors_by_mouse = []
            shuffles_by_mouse = []
            for session_type in session_types:
                errors_by_mouse.append(errors[session_type][age])
                shuffles_by_mouse.append(
                    np.nanmean(errors_shuffled[session_type][age], axis=1)
                )

            errors_by_mouse = np.vstack(errors_by_mouse)
            shuffles_by_mouse = np.vstack(shuffles_by_mouse)
            ax.plot(xlabels, errors_by_mouse, color=color, alpha=0.5)
            ax.plot(xlabels, shuffles_by_mouse, color="gray", alpha=0.5)
            errorfill(
                xlabels,
                np.nanmean(errors_by_mouse, axis=1),
                sem(errors_by_mouse, axis=1),
                ax=ax,
                color=color,
            )

            errorfill(
                xlabels,
                np.nanmean(shuffles_by_mouse, axis=1),
                sem(shuffles_by_mouse, axis=1),
                ax=ax,
                color="gray",
            )
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_xticklabels(xlabels, rotation=45, fontsize=16)
        axs[0].set_ylabel("Decoding error (cm)", fontsize=22)
        fig.tight_layout()

        errors_df = pd.DataFrame()
        for age in ages_to_plot:
            for session_type in session_types:
                errors_df[session_type] = errors[session_type][age]
            errors_df["mice"] = self.meta["grouped_mice"][age]
            errors_df["age"] = age

        errors_df = pd.melt(
            errors_df, id_vars=["mice", "age"], var_name="session", value_name="error"
        )

        return errors, errors_shuffled, errors_df, fig

    def multisession_spatial_decoder_anova(self, errors_df):
        anova_df = pg.rm_anova(errors_df, dv="error", within="session", subject="mice")
        pairwise_df = pg.pairwise_ttests(
            errors_df, dv="error", within="session", subject="mice", padjust="sidak"
        )

        return anova_df, pairwise_df

    def plot_spatial_decoder_errors(self, session_type, ages_to_plot="young", **kwargs):
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        errors = dict()
        for age in ages_to_plot:
            errors[age] = {"error": [], "errors_shuffled": []}
            for mouse in self.meta["grouped_mice"][age]:
                error, errors_shuffled = self.within_session_spatial_decoder(
                    mouse, session_type, **kwargs
                )
                errors[age]["error"].append(error)
                errors[age]["errors_shuffled"].append(errors_shuffled)

        fig, axs = plt.subplots(1, n_ages_to_plot)
        if n_ages_to_plot == 1:
            axs = [axs]
        for ax, age, color in zip(axs, ages_to_plot, plot_colors):
            boxes = ax.boxplot(
                errors[age]["errors_shuffled"],
                patch_artist=True,
                widths=0.75,
                zorder=0,
                showfliers=False,
            )
            color_boxes(boxes, "w")
            chance = np.nanmean([np.nanmean(e) for e in errors[age]["errors_shuffled"]])
            ax.text(1.1, chance - 5, "Chance", horizontalalignment="center", color="k")

            ax.scatter(
                np.arange(1, len(errors[age]["error"]) + 1),
                errors[age]["error"],
                c=color,
                marker="_",
                s=100,
                linewidth=5,
            )
            ax.set_xticklabels(self.meta["grouped_mice"][age], rotation=90)
            ax.axhline(y=chance, color="k")
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_xlabel("Mice")

        axs[0].set_ylabel("Decoder error (cm)")
        fig.tight_layout()

        return errors

    def test_classifier(
        self, X, y, kfolds=5, classifier=GaussianNB(), n_spatial_bins=125, shuffle=False
    ):
        skf = KFold(n_splits=kfolds)

        if shuffle:
            y = np.roll(y, np.random.randint(300, len(y)))

        errors = []
        for train, test in skf.split(X, y):
            classifier.fit(X[train], y[train])
            y_predicted = classifier.predict(X[test])

            error = np.nanmean(
                get_circular_error(y_predicted, y[test], n_spatial_bins=n_spatial_bins)
            )

            nbins_to_cm = 2 * np.pi / n_spatial_bins * 38.1  # radius of maze in cm
            errors.append(error.astype(float) * nbins_to_cm)

        return np.nanmean(errors)

    def plot_spatial_decoder(
        self,
        classifier=RandomForestClassifier(),
        session_pairs=(("Goals4", "Goals3"), ("Goals4", "Reversal")),
        n_spatial_bins=125,
        error_time_bin_size=300 * 15,
        predictors="cells",
        ages_to_plot=None,
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
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        fig, axs = plt.subplots(
            n_ages_to_plot, 2, sharex=True, sharey=True, figsize=(9, 7 * n_ages_to_plot)
        )
        if n_ages_to_plot == 1:
            axs = [axs]
        sessions = [session_pair[1] for session_pair in session_pairs]
        decoding_errors = dict()
        errors_, sessions_, mice_, ages_, time_bins_, = (
            [],
            [],
            [],
            [],
            [],
        )
        for age, cohort_axs, color in zip(ages_to_plot, axs, plot_colors):
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

            for session, session_ax in zip(sessions, cohort_axs):
                errors_arr = np.vstack(decoding_errors[age][session])
                m = np.nanmean(errors_arr, axis=0)
                se = sem(errors_arr, axis=0)
                errorfill(
                    np.linspace(0, 1, len(m)), m, yerr=se, color=color, ax=session_ax
                )

                [
                    session_ax.spines[side].set_visible(False)
                    for side in ["top", "right"]
                ]

        df = pd.DataFrame(
            {
                "mice": mice_,
                "age": ages_,
                "session_pairs": sessions_,
                "time_bins": time_bins_,
                "decoding_errors": errors_,
            }
        )
        fig.supylabel("Mean decoding error (cm)")
        fig.supxlabel("Time in session (normalized)")
        fig.tight_layout()

        return decoding_errors, df

    def spatial_decoder_anova(
        self,
        classifier=RandomForestClassifier(),
        session_pairs=(("Goals3", "Goals4"), ("Goals4", "Reversal")),
        n_spatial_bins=125,
        error_time_bin_size=300 * 15,
        predictors="cells",
    ):
        decoding_errors, df = self.plot_spatial_decoder(
            classifier=classifier,
            session_pairs=session_pairs,
            n_spatial_bins=n_spatial_bins,
            error_time_bin_size=error_time_bin_size,
            predictors=predictors,
        )

        anova_dfs = {
            age: pg.rm_anova(
                df.loc[df["age"] == age],
                dv="decoding_errors",
                within=["session_pairs", "time_bins"],
                subject="mice",
            )
            for age in ages
        }

        pairwise_dfs = {
            age: {
                session: pg.pairwise_ttests(
                    dv="decoding_errors",
                    within="time_bins",
                    subject="mice",
                    data=df[
                        np.logical_and(df["age"] == age, df["session_pairs"] == session)
                    ],
                    padjust="fdr_bh",
                )
                for session in np.unique(df["session_pairs"].values)
            }
            for age in ages
        }

        return anova_dfs, pairwise_dfs, df

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
        nbins_to_cm = 2 * np.pi / n_spatial_bins * 38.1  # radius of maze in cm

        d = d.astype(float) * nbins_to_cm
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
    def count_ensembles(
        self, grouped=True, normalize=False, sessions_to_plot=None, ages_to_plot=None
    ):
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
            sessions_to_plot = self.meta["session_types"]

        df = pd.DataFrame(index=self.meta["mice"])
        for session_type in sessions_to_plot:
            n_ensembles_sessions = []
            for mouse in self.meta["mice"]:
                session = self.data[mouse][session_type]
                n_ensembles_mouse = session.assemblies["significance"].nassemblies
                if normalize:
                    n_ensembles_mouse /= session.imaging["n_neurons"]

                n_ensembles_sessions.append(n_ensembles_mouse)

            df[session_type] = n_ensembles_sessions

        df["aged"] = [self.meta["aged"][mouse] for mouse in df.index]
        df = df.sort_values("aged")

        if grouped:
            # Make a dictionary that's [session_type][age] = list of ensemble counts.
            n_ensembles = dict()

            for session_type in sessions_to_plot:
                n_ensembles[session_type] = dict()

                for age in ages:
                    aged = age == "aged"
                    n_ensembles[session_type][age] = df[session_type].loc[
                        df["aged"] == aged
                    ]

            # Plot.
            ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
                ages_to_plot
            )
            fig, ax = plt.subplots()
            for age, color in zip(ages_to_plot, plot_colors):
                aged = True if age == "aged" else False
                n_neurons_plot = df.loc[df.aged == aged]
                n_neurons_plot = n_neurons_plot.drop(columns="aged")
                xticks = [
                    col.replace("Goals", "Training") for col in n_neurons_plot.columns
                ]
                ax.plot(xticks, n_neurons_plot.T, color=color, alpha=0.5)
                errorfill(
                    xticks,
                    n_neurons_plot.mean(axis=0),
                    yerr=sem(n_neurons_plot, axis=0),
                    ax=ax,
                    color=color,
                )
                plt.setp(ax.get_xticklabels(), rotation=45)
            ax.set_ylabel(ylabel, fontsize=22)
            ax.set_ylim([0, 100])
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]

            # self.set_age_legend(fig)
            fig.tight_layout()

            if ages_to_plot is None:
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
            ax.set_ylim([0, 100]) #Hard coded for Denise's aesthetic preference.
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            fig.subplots_adjust(bottom=0.2)

        return n_ensembles, df, fig

    def percent_spatial_ensembles(self, alpha=0.001):
        percent_spatial = pd.DataFrame()
        for mouse in self.meta["mice"]:
            for session in self.meta["session_types"]:
                pfs = self.data[mouse][session].assemblies["fields"].data

                spatial = np.where(pfs["spatial_info_pvals"] < alpha)[0]
                p = pd.DataFrame(
                    {
                        "age": "young"
                        if mouse not in self.meta["grouped_mice"]["aged"]
                        else "aged",
                        "mice": mouse,
                        "session_type": session,
                        "n_ensembles": [pfs["n_neurons"]],
                        "n_spatial": [len(spatial)],
                        "p_spatial": [len(spatial) / pfs["n_neurons"]],
                    }
                )
                percent_spatial = pd.concat((percent_spatial, p))

        return percent_spatial

    def plot_ensemble(self, mouse, session_type, ensemble_number):
        fig = self.data[mouse][session_type].plot_assembly(
            ensemble_number,
            members_only=False,
            z_score=True,
        )[0]

        return fig

    def plot_ex_patterns(self, mouse, session_type, pattern_numbers):
        assemblies = self.data[mouse][session_type].assemblies
        patterns = assemblies["patterns"][pattern_numbers]
        n_patterns = patterns.shape[0]
        bool_members, members = find_members(patterns)[:2]

        fig, axs = plt.subplots(
            n_patterns, 1, figsize=(6.4, 2 * n_patterns), sharey=True, sharex=True
        )
        for ax, pattern, members_, bool_members_ in zip(
            axs, patterns, members, bool_members
        ):
            non_members = np.where(~bool_members_.astype(bool))[0]
            markerlines, stemlines, _ = ax.stem(
                members_,
                pattern[members_],
                "c",
                markerfmt=" ",
                label="ensemble members",
            )
            plt.setp(markerlines, markersize=3)

            markerlines, stemlines, _ = ax.stem(
                non_members,
                pattern[non_members],
                "k",
                markerfmt=" ",
                label="non-members",
            )
            plt.setp(markerlines, markersize=3)
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]

        axs[-1].set_xlabel("Neurons", fontsize=22)
        fig.supylabel("Weights", fontsize=22)
        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels)
        fig.tight_layout()

        return fig

    def make_ensemble_raster(
        self, mouse, session_type, bin_size=0.6, running_only=False
    ):
        session = self.data[mouse][session_type]
        behavior_df = session.behavior.data["df"]
        lin_position = np.asarray(behavior_df["lin_position"])

        if running_only:
            running = session.spatial.data["running"]
        else:
            running = np.ones_like(session.spatial.data["running"], dtype=bool)

        filler = np.zeros_like(lin_position)
        activations = session.assemblies["activations"]

        if bin_size is None:
            bin_size = session.meta["spatial_bin_size"]

        bin_edges = spatial_bin(
            lin_position, filler, bin_size_cm=bin_size, nbins=None, one_dim=True
        )[1]

        rasters = nan_array(
            (
                session.assemblies["significance"].nassemblies,
                session.behavior.data["ntrials"],
                len(bin_edges) - 1,
            )
        )

        for trial_number in range(session.behavior.data["ntrials"]):
            time_bins = behavior_df["trials"] == trial_number
            positions_this_trial = behavior_df.loc[time_bins, "lin_position"]
            filler = np.zeros_like(positions_this_trial)
            running_this_trial = running[time_bins]

            for n, neuron in enumerate(activations):
                activation = neuron[time_bins] * running_this_trial
                rasters[n, trial_number, :] = spatial_bin(
                    positions_this_trial,
                    filler,
                    bins=bin_edges,
                    one_dim=True,
                    weights=activation,
                )[0]

        session.assemblies["fields"].data["rasters"] = rasters
        session.assemblies["fields"].data["tuning_curves"] = np.mean(rasters, axis=1)
        session.assemblies["fields"].meta["raster_bin_size"] = bin_size
        session.assemblies["fields"].meta["raster_running_only"] = running_only

    def plot_ensemble_raster(
        self, mouse, session_type, ensemble_number, bin_size=0.05, running_only=False
    ):
        session = self.data[mouse][session_type]
        try:
            assert (
                session.assemblies["fields"].meta["raster_bin_size"] == bin_size
                and session.assemblies["fields"].meta["raster_running_only"]
                == running_only
            ), "Stored raster parameters don't match inputs, recalculating rasters."

        except:
            self.make_ensemble_raster(
                mouse, session_type, bin_size=bin_size, running_only=running_only
            )
        rasters = session.assemblies["fields"].data["rasters"]

        behavior_data = session.behavior.data
        port_bins = find_reward_spatial_bins(
            behavior_data["df"]["lin_position"],
            np.asarray(behavior_data["lin_ports"]),
            spatial_bin_size_radians=bin_size,
        )[0]

        fig, ax = plt.subplots()
        im = ax.imshow(
            rasters[ensemble_number], cmap="viridis", interpolation="hanning"
        )
        # axs[1].plot(np.mean(rasters[ensemble_number], axis=0))

        for port in port_bins[behavior_data["rewarded_ports"]]:
            ax.axvline(x=port, color="w")

        ax.set_xticks(ax.get_xlim())
        ax.set_xticklabels([0, 220])

        ax.set_yticks([1, rasters[ensemble_number].shape[0] - 1])
        ax.axis("tight")
        ax.set_ylabel("Trial", fontsize=22)
        ax.set_xlabel("Linearized position (cm)", fontsize=22)
        cbar = fig.colorbar(
            im,
            ticks=[np.min(rasters[ensemble_number]), np.max(rasters[ensemble_number])],
        )
        cbar.ax.set_yticklabels([0, "Max"])
        cbar.set_label("Ensemble activity (a.u.)", rotation=270, fontsize=18)
        fig.tight_layout()

        return fig

    def scrollplot_ensemble_rasters(
        self, mouse, session_type, bin_size=0.2, running_only=False, subset=None
    ):

        session = self.data[mouse][session_type]
        behavior_data = session.behavior.data
        try:
            assert (
                session.assemblies["fields"].meta["raster_bin_size"] == bin_size
                and session.assemblies["fields"].meta["raster_running_only"]
                == running_only
            ), "Stored raster parameters don't match inputs, recalculating rasters."

        except:
            self.make_ensemble_raster(
                mouse, session_type, bin_size=bin_size, running_only=running_only
            )

        if subset is None:
            subset = range(session.assemblies["significance"].nassemblies)

        rasters = session.assemblies["fields"].data["rasters"][subset]
        tuning_curves = session.assemblies["fields"].data["tuning_curves"][subset]
        port_bins = find_reward_spatial_bins(
            behavior_data["df"]["lin_position"],
            np.asarray(behavior_data["lin_ports"]),
            spatial_bin_size_radians=bin_size,
        )[0]
        rewarded = behavior_data["rewarded_ports"]

        ensemble_labels = [f"Ensemble #{n}" for n in subset]
        self.raster_plot = ScrollPlot(
            plot_raster,
            nrows=2,
            rasters=rasters,
            tuning_curves=tuning_curves,
            port_bins=port_bins,
            rewarded=rewarded,
            cmap="plasma",
            interpolation="hanning",
            titles=ensemble_labels,
            figsize=(5, 8),
        )

        return self.raster_plot

    def plot_activation_trend(
        self,
        mouse,
        session_type,
        unit_number,
        x="trial",
        z_threshold=None,
        x_bin_size=1,
        data_type="ensembles",
        subset=None,
        alpha="sidak",
        ax=None,
    ):
        trends, binned_activations, slopes, _ = self.find_activity_trends(
            mouse,
            session_type,
            x=x,
            z_threshold=z_threshold,
            x_bin_size=x_bin_size,
            data_type=data_type,
            subset=subset,
            alpha=alpha,
        )
        n_xbins = binned_activations.shape[1]

        # Redoing some computations here, but too lazy to optimize.
        mk_result = mk.original_test(binned_activations[unit_number])
        x_pts = np.arange(n_xbins)
        trend_line = x_pts * mk_result.slope + mk_result.intercept

        if x_bin_size == 1:
            xlabel = "Trial"
        else:
            xlabel = "Trial block"
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        ax.scatter(x_pts, binned_activations[unit_number], c="m")
        ax.plot(x_pts, trend_line, color="k")
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel("Peak activation\nstrength (a.u.)", fontsize=22)
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        return fig

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
                    * 100
                    / n_neurons
                )
                proportion_unique_members[age].append(
                    len(np.unique(np.hstack(members))) / n_neurons
                )

        return proportion_unique_members, ensemble_size

    # def find_promiscuous_neurons(
    #     self, session_type, filter_method="sd", thresh=2, p_ensemble_thresh=0.1
    # ):
    #     p_promiscuous_neurons = {age: [] for age in ages}
    #     for age in ages:
    #         p_promiscuous_neurons[age] = []
    #
    #         for mouse in self.meta["grouped_mice"][age]:
    #             patterns = self.data[mouse][session_type].assemblies["patterns"]
    #             n_ensembles, n_neurons = patterns.shape
    #             memberships = find_memberships(
    #                 patterns, filter_method=filter_method, thresh=thresh
    #             )
    #
    #             ensemble_threshold = p_ensemble_thresh * n_ensembles
    #
    #             p_promiscuous_neurons[age].append(
    #                 sum(
    #                     np.asarray([len(ensemble) for ensemble in memberships])
    #                     > ensemble_threshold
    #                 )
    #                 / n_neurons
    #             )
    #
    #     return p_promiscuous_neurons

    def split_session_ensembles(self, mouse, session_type, overwrite_ensembles=False):
        """
        Split a session in half then look for ensembles separately for each half.

        """
        session = self.data[mouse][session_type]
        if session.meta["local"]:
            folder = get_equivalent_local_path(session.meta["folder"])
        else:
            folder = session.meta["folder"]
        fpath = os.path.join(folder, "SplitSessionEnsembles.pkl")

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
            split_data = np.array_split(
                processed_for_assembly_detection["processed"][0], 2, axis=1
            )

            split_ensembles = {
                half: find_assemblies(data, nullhyp="circ", plot=False, n_shuffles=500)
                for half, data in zip(["first", "second"], split_data)
            }

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
        self,
        show_plot=True,
        z_threshold=None,
        alpha="sidak",
        ages_to_plot=None,
    ):
        p_near, n_near = dict(), dict()
        port_types = ["rewarded", "previously_rewarded", "other", "relevant"]

        for age in ages:
            p_near[age] = {key: [] for key in port_types}
            n_near[age] = {key: [] for key in port_types}
            for mouse in self.meta["grouped_mice"][age]:
                # Find the closest ports for each ensemble.
                closest_ports = self.find_ensemble_port_locations(mouse, "Reversal")[1]

                trends = self.find_activity_trends(
                    mouse,
                    "Reversal",
                    data_type="ensembles",
                    z_threshold=z_threshold,
                    alpha=alpha,
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
                        n = np.sum(
                            i in port_locations[port_type]
                            for i in closest_ports[trends["decreasing"]]
                        )
                        p_near[age][port_type].append(n / len(trends["decreasing"]))

                        n_near[age][port_type].append(n)

                except ZeroDivisionError:
                    for port_type in port_types:
                        p_near[age][port_type].append(0)
                        n_near[age][port_type].append(0)

        if show_plot:
            ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
                ages_to_plot
            )
            fig, axs = plt.subplots(
                1, n_ages_to_plot, sharey=True, figsize=(6 * n_ages_to_plot, 6)
            )
            if n_ages_to_plot == 1:
                axs = [axs]

            for ax, age in zip(axs, ages_to_plot):
                bottom = np.zeros_like(n_near[age]["rewarded"])

                mice = self.meta["grouped_mice"][age]
                for p, label, c in zip(
                    [
                        n_near[age][key]
                        for key in ["rewarded", "previously_rewarded", "other"]
                    ],
                    ["Currently rewarded", "Previously rewarded", "Never rewarded"],
                    ["steelblue", "dimgray", "w"],
                ):
                    ax.bar(mice, p, bottom=bottom, label=label, color=c, edgecolor="k")
                    bottom += p

                ax.set_xticklabels([m[0] for m in mice], rotation=90)
                ax.set_xlabel("Mouse", fontsize=22)
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]

            axs[0].set_ylabel("# fading ensembles", fontsize=22)
            axs[-1].legend(bbox_to_anchor=(1.05, 1), loc="center", fontsize=14)
            fig.tight_layout()
        else:
            fig = None

        obs_freq = dict()
        exp_freq = dict()
        chi_result = dict()
        for age in ages:
            obs_freq[age] = np.vstack(
                (
                    n_near[age]["rewarded"],
                    n_near[age]["previously_rewarded"],
                    n_near[age]["other"],
                )
            )

            exp_freq[age] = np.sum(obs_freq[age], axis=0) * np.repeat(
                np.asarray([0.25, 0.25, 0.5])[:, np.newaxis],
                obs_freq[age].shape[1],
                axis=1,
            )

            chi_result[age] = chisquare(obs_freq[age], exp_freq[age], axis=0)

        return p_near, n_near, chi_result, fig

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

    def fit_lick_decoder(
        self,
        mouse: str,
        session_pair: tuple,
        classifier=RandomForestClassifier,
        licks_to_include="all",
        lag=0,
        fps=15,
        data_type="ensembles",
        exclude=None,
        do_zscore=True,
        shuffle=False,
        **classifier_kwargs,
    ):
        split_session = len(np.unique(session_pair)) == 1
        # If the specified sessions in session_pair are the same, split the session in half
        # and train on the first half.
        if split_session:
            if data_type == "ensembles":
                raise NotImplementedError(f"{data_type} not yet implemented.")
            elif data_type in ["S", "S_binary"]:
                registered_data = np.array_split(
                    self.data[mouse][session_pair[0]].imaging[data_type], 2, axis=1
                )
            else:
                raise NotImplementedError(f"{data_type} not implemented.")

        # Otherwise, register the sessions' neurons or ensembles.
        else:
            if data_type == "ensembles":
                registered_ensembles = self.match_ensembles(mouse, session_pair)
                registered_data = registered_ensembles["matched_activations"]
            elif data_type in ["S", "S_binary"]:
                registered_data = self.rearrange_neurons(mouse, session_pair, data_type)
            else:
                raise NotImplementedError(f"{data_type} not implemented.")

        if data_type == "S_binary":
            registered_data = [np.asarray(data, dtype=int) for data in registered_data]

        if exclude is None:
            exclude = np.zeros((registered_data[0].shape[0],), dtype=bool)
        if data_type == "ensembles":
            exclude = np.logical_or(exclude, registered_ensembles["poor_matches"])

        if split_session:
            lick_data = np.array_split(
                np.asarray(
                    self.data[mouse][session_pair[0]].behavior.data["df"]["lick_port"]
                ),
                2,
            )
        else:
            lick_data = [
                np.asarray(
                    self.data[mouse][session_type].behavior.data["df"]["lick_port"]
                )
                for session_type in session_pair
            ]

        # Gather predictor neural data.
        X = dict()
        y = dict()
        for session_activations, licks, t, session_type in zip(
            registered_data,
            lick_data,
            ["train", "test"],
            session_pair,
        ):
            activations = session_activations[~exclude].T
            if do_zscore:
                activations = zscore(activations, axis=1)

            imp = SimpleImputer(
                missing_values=np.nan, strategy="constant", fill_value=0
            )
            activations = imp.fit_transform(activations)

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

        clf = classifier(**classifier_kwargs)
        clf.fit(X["train"], y["train"])

        if shuffle:
            shuffled_clfs = [
                classifier(**classifier_kwargs).fit(
                    X["train"], np.random.permutation(y["train"])
                )
                for i in range(50)
            ]
        else:
            shuffled_clfs = []

        return X, y, clf, shuffled_clfs

    def compare_lick_decoding_accuracy(
        self,
        session_pairs=(("Goals4", "Goals3"), ("Goals4", "Reversal")),
        classifier=RandomForestClassifier,
        data_type="S",
        do_zscore=False,
        licks_to_include="first",
        **classifier_kwargs,
    ):
        """
        Compare the lick decoding accuracy of classifiers trained and tested on different sessions.

        :parameters
        ---
        session_pairs: tuple of tuples ((session1, session2), (session3, session4))
            Train on session 1 and session 3, test on session 2 and session 4.


        """
        accuracy = dict()
        for age in ages:
            accuracy[age] = dict()
            for session_pair in session_pairs:
                accuracy[age][session_pair] = []

                for mouse in self.meta["grouped_mice"][age]:
                    X, y, clf, _ = self.fit_lick_decoder(
                        mouse,
                        session_pair,
                        classifier=classifier,
                        data_type=data_type,
                        do_zscore=do_zscore,
                        licks_to_include=licks_to_include,
                        **classifier_kwargs,
                    )

                    accuracy[age][session_pair].append(clf.score(X["test"], y["test"]))

        fig, axs = plt.subplots(1, 2, sharey=True)
        labels = [
            f"Train: {session_pair[0]}\nTest: {session_pair[1]}"
            for session_pair in session_pairs
        ]
        for ax, age, color in zip(axs, ages, age_colors):
            ax.plot(
                labels,
                [accuracy[age][session_pair] for session_pair in session_pairs],
                color=color,
            )
            ax.set_title(age)
            ax.set_xlim([-0.5, 1.5])
            plt.setp(ax.get_xticklabels(), rotation=45)
        axs[0].set_ylabel("Lick decoder accuracy")

        return accuracy

    def correlate_fade_slope_to_learning_slope(
        self,
        age_to_plot="young",
        session_type="Reversal",
        performance_metric="CRs",
        dv="reg_slope",
    ):
        c = age_colors[ages.index(age_to_plot)]
        df = self.learning_rate(session_type, performance_metric=performance_metric)
        for age in ages:
            for mouse in self.meta["grouped_mice"][age]:
                trends, _, slopes, tau = self.find_activity_trends(
                    mouse,
                    session_type,
                    x="trial",
                    z_threshold=None,
                    x_bin_size=1,
                    data_type="ensembles",
                    subset=None,
                    alpha="sidak",
                )

                mean_slope = np.mean(slopes)
                df.loc[df["mice"] == mouse, "ensemble_slope"] = mean_slope
                df.loc[df["mice"] == mouse, "age"] = age

        performance_label = {
            "hits": "hit rate",
            "CRs": "corr. rej. rate",
            "d_prime": "d'",
        }
        ylabel = {
            "reg_slope": f"Regression slope of\n"
            f"{performance_label[performance_metric]} over trials",
            "peak_trial": f"Trial @ "
            f"{performance_label[performance_metric]}\nasymptote",
            "norm_peak_trial": f"Time of peak\n"
            f"{performance_label[performance_metric]}, normalized",
        }
        fig, ax = plt.subplots()
        x = df.loc[df["age"] == age_to_plot, "ensemble_slope"]
        y = df.loc[df["age"] == age_to_plot, dv]
        ax.scatter(x, y, edgecolors="k", s=100, c=c)
        z = np.polyfit(x, y, 1)
        y_hat = np.poly1d(z)(x)
        ax.plot(x, y_hat, color=c)
        ax.set_xlabel("Mean ensemble trend slope", fontsize=22)
        ax.set_ylabel(f"Learning rate\n({ylabel[dv]})", fontsize=22)
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()
        print(spearmanr(x, y, nan_policy="omit"))

        return df, fig

    def ensemble_behavior_causality(
        self, mouse, window=6, strides=2, performance_metric="CRs", func=np.mean
    ):
        session = self.data[mouse]["Reversal"].behavior
        session.sdt_trials(
            rolling_window=window,
            trial_interval=strides,
            plot=False,
        )
        df = pd.DataFrame(
            {
                "t": range(len(session.sdt[performance_metric])),
                "dv": session.sdt[performance_metric],
            }
        )

        trends, binned_activations = self.find_activity_trends(
            mouse,
            "Reversal",
            x="trial",
            z_threshold=None,
            x_bin_size=1,
            alpha="sidak",
            data_type="ensembles",
        )[:2]

        n_ensembles = binned_activations.shape[0]
        binned_activations = binned_activations.T
        blocked = np.squeeze(
            sliding_window_view(
                binned_activations, (window, binned_activations.shape[1]), axis=(0, 1)
            )[::strides]
        )
        ensemble_activity = np.vstack(
            [func(trial_block, axis=0) for trial_block in blocked]
        )

        df = pd.concat((df, pd.DataFrame(ensemble_activity)), axis=1)

        df_diff = df.diff().dropna()
        test = "ssr_chi2test"
        maxlag = 4
        granger_df = pd.DataFrame(
            nan_array((len(trends["decreasing"]), 2)), index=trends["decreasing"]
        )
        for ensemble_number in trends["decreasing"]:
            result = adfuller(df_diff[ensemble_number])[1]
            if result < 0.05:
                g = grangercausalitytests(
                    df_diff[["dv", ensemble_number]], maxlag=maxlag, verbose=False
                )
                p = np.min([round(g[i + 1][0][test][1], 4) for i in range(maxlag)])
                granger_df.loc[ensemble_number, 0] = p

                g = grangercausalitytests(
                    df_diff[[ensemble_number, "dv"]], maxlag=maxlag, verbose=False
                )
                p = np.min([round(g[i + 1][0][test][1], 4) for i in range(maxlag)])
                granger_df.loc[ensemble_number, 1] = p

        return granger_df

    def lick_decoder(
        self,
        mouse,
        session_types,
        classifier=RandomForestClassifier,
        licks_to_include="all",
        n_splits=6,
        lag=0,
        show_plot=True,
        ax=None,
        data_type="ensembles",
        shuffle=False,
        **classifier_kwargs,
    ):
        X, y, classifier, shuffled_clfs = self.fit_lick_decoder(
            mouse,
            session_types,
            classifier=classifier,
            lag=lag,
            licks_to_include=licks_to_include,
            data_type=data_type,
            shuffle=shuffle,
            **classifier_kwargs,
        )

        X_split = np.array_split(X["test"], n_splits)
        y_split = np.array_split(y["test"], n_splits)
        scores = [
            classifier.score(X_chunk, y_chunk)
            for X_chunk, y_chunk in zip(X_split, y_split)
        ]

        chance = []
        if shuffle:
            for surrogate in shuffled_clfs:
                chance.append(
                    [
                        surrogate.score(X_chunk, y_chunk)
                        for X_chunk, y_chunk in zip(X_split, y_split)
                    ]
                )
            chance = np.vstack(chance)

        if show_plot:
            x = np.arange(0, 31, 30 / n_splits).astype(int)
            xtick_labels = [f"{i}-{j} min" for i, j in zip(x[:-1], x[1:])]
            if ax is None:
                fig, ax = plt.subplots()

            ax.plot(xtick_labels, scores, "k", alpha=0.2)
            # ax.plot(xtick_labels, chance.T, "r", alpha=0.1)

        return scores, chance

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
        X, y, classifier, _ = self.fit_lick_decoder(
            mouse,
            session_types,
            classifier=classifier,
            lag=lag,
            licks_to_include=licks_to_include,
            data_type="ensembles",
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

    def plot_lick_decoder(
        self,
        classifier=RandomForestClassifier,
        licks_to_include="first",
        n_splits=6,
        session_types=(("Goals4", "Goals3"), ("Goals4", "Reversal")),
        lag=0,
        ages_to_plot=None,
        data_type="ensembles",
        shuffle=False,
        **classifier_kwargs,
    ):
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        scores = dict()
        chance = {age: dict() for age in ages_to_plot}
        fig, axs = plt.subplots(
            n_ages_to_plot,
            len(session_types),
            sharey=True,
            sharex=True,
            figsize=(9, 7 * n_ages_to_plot),
        )
        if n_ages_to_plot == 1:
            axs = [axs]

        time_bins = np.arange(n_splits)
        mice_, decoded_session, ages_, time_bins_ = [], [], [], []
        for age, cohort_ax in zip(ages_to_plot, axs):
            scores[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                for ax, session_pair in zip(cohort_ax, session_types):
                    chance[age][session_pair] = []
                    scores_, chance_ = self.lick_decoder(
                        mouse,
                        session_pair,
                        classifier=classifier,
                        licks_to_include=licks_to_include,
                        n_splits=n_splits,
                        show_plot=True,
                        lag=lag,
                        ax=ax,
                        data_type=data_type,
                        shuffle=shuffle,
                        **classifier_kwargs,
                    )

                    scores[age].extend(scores_)
                    chance[age][session_pair].extend(chance_)
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
                "scores": np.hstack([scores[age] for age in ages_to_plot]),
            }
        )
        means = df.groupby(["age", "decoded_session", "time_bins"]).mean()
        sems = df.groupby(["age", "decoded_session", "time_bins"]).sem()

        x = np.arange(0, 31, 30 / n_splits).astype(int)
        xtick_labels = [f"{i}-{j} min" for i, j in zip(x[:-1], x[1:])]
        for age, row_ax, color in zip(ages_to_plot, axs, plot_colors):
            for session_pair, ax in zip(session_types, row_ax):
                errorfill(
                    xtick_labels,
                    np.squeeze(means.loc[age].loc[session_pair[1]].values),
                    yerr=np.squeeze(sems.loc[age].loc[session_pair[1]].values),
                    color=color,
                    ax=ax,
                )

                if shuffle:
                    errorfill(
                        xtick_labels,
                        np.nanmean(chance[age][session_pair], axis=0),
                        yerr=sem(chance[age][session_pair], axis=0),
                        color="gray",
                        ax=ax,
                    )
                else:
                    ax.axhline(y=1 / 8, color="darkred")
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                ax.set_title(
                    f"trained on {session_pair[0].replace('Goals', 'Training')}"
                    f"\n tested on {session_pair[1].replace('Goals', 'Training')}"
                )
                ax.set_ylim([0, 1])

                [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.supylabel("Lick port decoding accuracy")
        fig.tight_layout()

        return df, fig

    def lick_anova(
        self,
        classifier=RandomForestClassifier,
        licks_to_include="first",
        n_splits=6,
        session_types=(("Goals4", "Goals3"), ("Goals4", "Reversal")),
        lag=0,
        data_type="ensembles",
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
        df = self.plot_lick_decoder(
            classifier=classifier,
            licks_to_include=licks_to_include,
            n_splits=n_splits,
            session_types=session_types,
            lag=lag,
            ages_to_plot=None,
            data_type=data_type,
            **classifier_kwargs,
        )[0]

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

    # def ensemble_feature_selector(
    #     self,
    #     mouse,
    #     session_type,
    #     licks_only=True,
    #     feature_selector=RFECV,
    #     estimator=GaussianNB(),
    #     show_plot=True,
    #     **kwargs,
    # ):
    #     session = self.data[mouse][session_type]
    #     ensemble_activations = zscore(session.assemblies["activations"], axis=0).T
    #     licks = np.asarray(session.behavior.data["df"]["lick_port"])
    #
    #     if licks_only:
    #         licking = licks > -1
    #         ensemble_activations = ensemble_activations[licking]
    #         licks = licks[licking]
    #
    #     fs = feature_selector(
    #         estimator=estimator, cv=StratifiedKFold(3), n_jobs=6, **kwargs
    #     )
    #     fs.fit(ensemble_activations, licks)
    #
    #     # if show_plot:
    #     #     fig, ax = plt.subplots()
    #     #     ax.set_xlabel("Number of features selected")
    #     #     ax.set_ylabel("Cross validation score (nb of correct classifications)")
    #     #     ax.plot(fs.grid_scores_)
    #
    #     return fs

    def plot_ensemble_sizes(
        self,
        filter_method="sd",
        thresh=2,
        data_type="unique_members",
        ages_to_plot=None,
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
        ensemble_size_df = pd.DataFrame()
        for session_type in self.meta["session_types"]:
            (
                proportion_unique_members[session_type],
                ensemble_size[session_type],
            ) = self.count_unique_ensemble_members(
                session_type, filter_method=filter_method, thresh=thresh
            )

            for age in ages:
                size_dict = pd.DataFrame(
                    {
                        "mice": self.meta["grouped_mice"][age],
                        "age": age,
                        "session": session_type,
                        "ensemble_size": ensemble_size[session_type][age],
                    }
                )
                ensemble_size_df = pd.concat((ensemble_size_df, size_dict))

        data = {
            "unique_members": proportion_unique_members,
            "ensemble_size": ensemble_size,
        }
        ylabels = {
            "unique_members": "Unique members / total # neurons",
            "ensemble_size": "% of total neurons per ensemble",
        }

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        fig, ax = plt.subplots()
        for age, color in zip(ages_to_plot, plot_colors):
            piv_df = ensemble_size_df[ensemble_size_df.age == age].pivot(
                index="mice", columns="session", values="ensemble_size"
            )

            xticks = [col.replace("Goals", "Training") for col in piv_df.columns]
            ax.plot(xticks, piv_df.T, color=color, alpha=0.5)
            errorfill(
                xticks,
                piv_df.mean(axis=0),
                yerr=sem(piv_df, axis=0),
                ax=ax,
                color=color,
            )
            plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel(ylabels[data_type], fontsize=22)
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        return data, ensemble_size_df, fig

    def ensemble_size_anova(self, ensemble_size_df, age="young"):
        anova_df = pg.rm_anova(
            ensemble_size_df[ensemble_size_df["age"] == age],
            dv="ensemble_size",
            within="session",
            subject="mice",
        )

        pairwise_df = pg.pairwise_ttests(
            ensemble_size_df[ensemble_size_df["age"] == age], correction="sidak"
        )

        return anova_df, pairwise_df

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

    # def plot_lapsed_assemblies(self, mouse, session_types, detected="everyday"):
    #     lapsed_assemblies = self.get_lapsed_assembly_activation(
    #         mouse, session_types, detected=detected
    #     )
    #     spiking = self.rearrange_neurons(
    #         mouse, session_types, "spike_times", detected=detected
    #     )
    #
    #     n_sessions = len(lapsed_assemblies["activations"])
    #     for i, pattern in enumerate(lapsed_assemblies["patterns"]):
    #         fig, axs = plt.subplots(n_sessions, 1)
    #         for ax, activation, spike_times in zip(
    #             axs, lapsed_assemblies["activations"], spiking
    #         ):
    #             plot_assembly(pattern, activation[i], spike_times, ax=ax)
    #
    #     return lapsed_assemblies, spiking

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
            raise NotImplementedError

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
        x_bin_size=1,
        data_type="ensembles",
        subset=None,
        alpha="sidak",
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
            raise NotImplementedError
        slopes, tau = nan_array(data.shape[0]), nan_array(data.shape[0])
        if subset is not None:
            data = data[subset]

        # Binarize activations so that there are no strings of 1s spanning multiple frames.
        activations = np.zeros_like(data)
        if z_threshold is not None:
            for i, unit in enumerate(data):
                # on_frames = contiguous_regions(unit > z_threshold)[:, 0]
                # activations[i, on_frames] = 1
                activations[i, unit > z_threshold] = unit[unit > z_threshold]
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
                binned_activations[:, i] = np.nanmax(activations[:, in_trial], axis=1)

        elif x is None:
            binned_activations = activations

        else:
            raise NotImplementedError("Invalid value for x.")

        # Group cells/ensembles into either increasing, decreasing, or no trend in occurrence rate.
        trends = {key: [] for key in ["no trend", "decreasing", "increasing"]}

        if type(alpha) is str:
            pvals = []
            pval_thresh = 0.005
            for i, unit in enumerate(binned_activations):
                mk_test = mk.original_test(unit, alpha=0.05)
                pvals.append(mk_test.p)

                slopes[i] = mk_test.slope
                tau[i] = mk_test.Tau

            corr_pvals = multipletests(pvals, alpha=pval_thresh, method=alpha)[1]

            for i, (corr_pval, slope) in enumerate(zip(corr_pvals, slopes)):
                if corr_pval < pval_thresh:
                    direction = "increasing" if slope > 0 else "decreasing"
                    trends[direction].append(i)
                else:
                    trends["no trend"].append(i)
        else:
            if subset is not None:
                for i, unit in zip(subset, binned_activations):
                    mk_test = mk.original_test(unit, alpha=alpha)
                    trends[mk_test.trend].append(i)
                    slopes[i] = mk_test.slope
                    tau[i] = mk_test.Tau
            else:
                for i, unit in enumerate(binned_activations):
                    mk_test = mk.original_test(unit, alpha=alpha)
                    trends[mk_test.trend].append(i)
                    slopes[i] = mk_test.slope
                    tau[i] = mk_test.Tau

        return trends, binned_activations, slopes, tau

    # def compare_trend_slopes(
    #     self, mouse, session_pair=("Goals4", "Reversal"), metric='slopes', **kwargs
    # ):
    #     trends = dict()
    #     slopes = dict()
    #     taus = dict()
    #
    #     registered_ensembles = self.match_ensembles(mouse, session_pair)
    #     for session_type in session_pair:
    #         trends[session_type], _, slopes[session_type], taus[session_type] = \
    #             self.find_activity_trends(
    #                 mouse, session_type, **kwargs
    #             )
    #
    #     fading_ensembles = np.asarray(trends[session_pair[0]]["decreasing"])
    #     if fading_ensembles.size == 0:
    #         return [], []
    #     reference_ensembles = fading_ensembles[
    #         ~registered_ensembles["poor_matches"][fading_ensembles]
    #     ]
    #     fading_ensembles_yesterday = registered_ensembles["matches"][
    #         reference_ensembles
    #     ]
    #
    #     if metric == 'slopes':
    #         x = slopes[session_pair[0]][fading_ensembles]
    #         y = slopes[session_pair[1]][fading_ensembles_yesterday]
    #     else:
    #         x = taus[session_pair[0]][fading_ensembles]
    #         y = taus[session_pair[1]][fading_ensembles_yesterday]
    #
    #     return x, y
    #
    # def compare_all_fading_ensemble_slopes(self, **kwargs):
    #     reversal_slopes = []
    #     training4_slopes = []
    #     fig, ax = plt.subplots()
    #     for age, color in zip(ages, age_colors):
    #         for mouse in self.meta["grouped_mice"][age]:
    #             x, y = self.compare_trend_slopes(mouse, **kwargs)
    #             ax.scatter(x, y, edgecolor=color)
    #
    #             ax.set_xlabel("Ensemble activation slope on Reversal")
    #             ax.set_ylabel("Ensemble activation slope on Training4")
    #
    #             reversal_slopes.extend(x)
    #             training4_slopes.extend(y)
    #
    #     ax.axis("equal")
    #
    #     return reversal_slopes, training4_slopes

    def compare_ensemble_trends(
        self, mouse, session_pair=("Goals4", "Reversal"), **kwargs
    ):
        metrics = {session_type: {} for session_type in session_pair}

        for session_type in session_pair:
            (
                metrics[session_type]["slopes"],
                metrics[session_type]["taus"],
            ) = self.find_activity_trends(mouse, session_type, **kwargs)[2:]

        return metrics

    def proportion_ensemble_trends(self, mouse, session_type, ax=None, **kwargs):
        trends = self.find_activity_trends(mouse, session_type, **kwargs)[0]
        trend_types = ["no trend", "increasing", "decreasing"]
        if ax is None:
            fig, ax = plt.subplots()

        sizes = [len(trends[trend]) for trend in trend_types]
        ax.pie(sizes, labels=trend_types)

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
        x_bin_size=1,
        z_threshold=None,
        alpha="sidak",
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
                    alpha=alpha,
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
        session_types=None,
        x="trial",
        x_bin_size=1,
        z_threshold=None,
        data_type="ensembles",
        show_plot=True,
        alpha=0.01,
    ):
        if session_types is None:
            session_types = self.meta["session_types"]

        session_labels = [label.replace("Goals", "Training") for label in session_types]
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

    def plot_proportion_changing_units(
        self,
        x="trial",
        x_bin_size=1,
        z_threshold=None,
        sessions=["Goals4", "Reversal"],
        trend="decreasing",
        data_type="ensembles",
        alpha="sidak",
        show_plot=True,
        ages_to_plot=None,
    ):
        ensemble_trends, ensemble_counts = self.plot_assembly_trends(
            session_types=sessions,
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
            ylabel_trend = {
                "decreasing": "remodeling",
                "increasing": "rising",
                "no trend": "flat",
            }
            ylabel_dtype = {"S": "cells", "ensembles": "ensembles"}

            ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
                ages_to_plot
            )

            fig, axs = plt.subplots(
                1, n_ages_to_plot, figsize=(3.5 * n_ages_to_plot, 6), sharey=True
            )
            if n_ages_to_plot > 1:
                fig.subplots_adjust(wspace=0)
            if n_ages_to_plot == 1:
                axs = [axs]
            for i, (ax, age, color) in enumerate(zip(axs, ages_to_plot, plot_colors)):
                # boxes = ax.boxplot(
                #     p_changing_split_by_age[age],
                #     patch_artist=True,
                #     widths=0.75,
                #     zorder=0,
                #     showfliers=False,
                # )
                # color_boxes(boxes, [color, color])
                ax.bar(
                    [0, 1],
                    np.mean(p_changing_split_by_age[age], axis=1),
                    yerr=sem(
                        np.asarray(p_changing_split_by_age[age]).astype(float), axis=1
                    ),
                    color=color,
                    linewidth=1,
                    ecolor="k",
                    edgecolor="k",
                    zorder=1,
                    error_kw=dict(lw=1, capsize=5, capthick=1, zorder=0),
                )

                data_points = np.vstack(p_changing_split_by_age[age]).T
                for mouse_data in data_points:
                    ax.plot(
                        jitter_x([0, 1], 0.05),
                        mouse_data,
                        "o-",
                        color="k",
                        markerfacecolor=color,
                        zorder=2,
                        markersize=10,
                        linewidth=1,
                    )

                if i > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(
                        f"Proportion {ylabel_trend[trend]} {ylabel_dtype[data_type]}",
                        fontsize=22,
                    )

                if n_ages_to_plot == 2:
                    ax.set_title(age)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(
                    [
                        session_type.replace("Goals", "Training")
                        for session_type in sessions
                    ],
                    rotation=45,
                    fontsize=16,
                )
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            fig.tight_layout()
        else:
            fig = None

        return p_changing_split_by_age, fig

    def hist_ensemble_taus(self, mouse, session_type, ax=None):
        tau = self.find_activity_trends(mouse, session_type)[3]

        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(tau)

        return tau

    def make_fading_ensemble_df(self, p_changing_split_by_age):
        p = p_changing_split_by_age
        df = pd.concat(
            [
                pd.DataFrame(
                    np.asarray(p[age]).T,
                    index=self.meta["grouped_mice"][age],
                    columns=[str(x.session.values) for x in p[age]],
                )
                for age in ages
            ]
        )

        df["aged"] = [self.meta["aged"][mouse] for mouse in df.index]

        return df

    def correlate_prop_changing_ensembles_to_behavior(
        self,
        x="trial",
        x_bin_size=1,
        z_threshold=None,
        trend="decreasing",
        performance_metric="CRs",
        data_type="ensembles",
        window=5,
        alpha="sidak",
        ax=None,
        ages_to_plot=None,
    ):
        ensemble_trends, ensemble_counts = self.plot_assembly_trends(
            session_types=["Reversal"],
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

        performance = self.performance_session_type(
            "Reversal",
            window=window,
            performance_metric=performance_metric,
        )

        if ax is None:
            fig, ax = plt.subplots()

        ylabels = {
            "CRs": "Peak correct rejection rate",
            "hits": "Peak hit rate",
            "d_prime": "Peak d'",
        }

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        for age, c in zip(ages_to_plot, plot_colors):
            prop_fading = np.asarray(p_changing_split_by_age[age], dtype=float)
            perf = performance[age]
            ax.scatter(
                prop_fading,
                perf,
                facecolors=c,
                edgecolors="k",
                s=100,
            )
            z = np.polyfit(prop_fading, perf, 1)
            y_hat = np.poly1d(z)(prop_fading)
            ax.plot(prop_fading, y_hat, color=c)

        ax.set_xlabel("Proportion remodeling ensembles", fontsize=22)
        ax.set_ylabel(ylabels[performance_metric], fontsize=22)
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]

        if n_ages_to_plot == 2:
            ax.legend(ages_to_plot, loc="lower right")

        fig.tight_layout()

        for age in ages_to_plot:
            r, pvalue = spearmanr(
                p_changing_split_by_age[age], performance[age], alternative="greater"
            )
            msg = f"{age} r = {np.round(r, 3)}, p = {np.round(pvalue, 4)}"
            if pvalue < 0.05:
                msg += "*"
            print(msg)

        p_changing = np.hstack([p_changing_split_by_age[age] for age in ages])
        perf = np.hstack([performance[age] for age in ages])

        if ages_to_plot == ages:
            r, pvalue = spearmanr(p_changing, perf, alternative="greater")
            msg = f"Young and aged together r = {np.round(r, 3)}, p = {np.round(pvalue, 3)}"
            if pvalue < 0.05:
                msg += "*"
            print(msg)

        return p_changing_split_by_age, performance, fig

    def make_corr_df(self, p_changing_split_by_age, performance):
        p = p_changing_split_by_age
        df = pd.concat(
            [
                pd.DataFrame(
                    np.asarray(p[age]).T,
                    index=self.meta["grouped_mice"][age],
                    columns=["proportion fading ensembles"],
                )
                for age in ages
            ]
        )

        df["performance"] = np.hstack([performance[age] for age in ages])
        df["aged"] = [self.meta["aged"][mouse] for mouse in df.index]

        return df

    def match_ensembles(self, mouse, session_pair):
        """
        Match assemblies across two sessions. For each assembly in the first session of the session_types tuple,
        find the corresponding assembly in the second session by taking the highest cosine similarity between two
        assembly patterns.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_pair: tuple
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
        split_session = True if len(np.unique(session_pair)) == 1 else False
        if split_session:
            split_ensembles = self.split_session_ensembles(mouse, session_pair[0])
            rearranged_patterns = [
                split_ensembles[half]["patterns"] for half in ["first", "second"]
            ]
            patterns_iterable = rearranged_patterns.copy()
            rearranged_patterns = [pattern.T for pattern in rearranged_patterns]

            activations = [
                split_ensembles[half]["activations"] for half in ["first", "second"]
            ]
        else:
            # Get the patterns from each session, matching the neurons.
            rearranged_patterns = self.rearrange_neurons(
                mouse, session_pair, data_type="patterns", detected="everyday"
            )

            # To keep consistent with its other outputs, self.rearrange_neurons()
            # gives a neuron x something (in this use case, assemblies) matrix.
            # We actually want to iterate over assemblies here, so transpose.
            patterns_iterable = [
                session_patterns.T for session_patterns in rearranged_patterns
            ]
            activations = [
                self.data[mouse][session].assemblies["activations"]
                for session in session_pair
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
        order = np.argsort(registered_patterns[0])

        fig = plt.figure(figsize=(14, 9))
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
            ["teal", "navy"],
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
                activation_color=c,
            )
            ax.set_title(session.replace("Goals", "Training"), fontsize=22, color=c)
            ax.set_rasterized(True)
            # Plot the patterns.
            pattern_ax = plot_pattern(
                pattern, ax=pattern_ax, color=c, alpha=0.5, order=order
            )
            plt.setp(ax.spines.values(), linewidth=2)

        # Label the patterns.
        pattern_ax.legend(
            [session.replace("Goals", "Training") for session in session_types],
            fontsize=22,
        )
        title = f"Cosine similarity: {np.round(similarity, 3)}"
        if poor_match:
            title += " NON-SIG MATCH!"
        pattern_ax.set_title(title, fontsize=22)
        pattern_ax.set_xticks([0, registered_patterns[0].shape[0]])
        pattern_ax = beautify_ax(pattern_ax)
        fig.tight_layout()

        return fig

    def plot_registered_patterns(
        self,
        patterns,
        ensemble_number,
        same_plot=False,
        do_sort=True,
        sort_by=0,
        colors=["k", "k"],
    ):
        if same_plot:
            fig, ax = plt.subplots()
            axs = [ax, ax]
        else:
            fig, axs = plt.subplots(2, 1, figsize=(3.5, 5))

        if do_sort:
            order = np.argsort(patterns[sort_by, ensemble_number])
        else:
            order = range(patterns.shape[-1])

        for i, (ax, color) in enumerate(zip(axs, colors)):
            plot_pattern(
                patterns[i, ensemble_number][order],
                ax=ax,
                alpha=0.5,
                markersize=3,
                color=color,
            )
            ax.axis("off")
        fig.tight_layout()

        return fig, axs, order

    def plot_ensemble_registration_ex(
        self, mouse="Miranda", session_types=("Goals3", "Goals4"), ensemble_number=11
    ):
        registered_ensembles = self.match_ensembles(mouse, session_types)
        patterns = registered_ensembles["matched_patterns"]
        order = self.plot_registered_patterns(patterns, ensemble_number)[2]

        n = 3
        non_matches = [
            registered_ensembles["matches"][ensemble_number + i] for i in range(n)
        ]
        fig, axs = plt.subplots(n, 1, figsize=(3.5, 2.5 * n))

        for non_match, ax in zip(non_matches, axs):
            plot_pattern(
                patterns[1, non_match][order], ax=ax, alpha=0.5, markersize=3, color="k"
            )
            ax.axis("off")
        fig.tight_layout()

    def plot_matched_ensembles(self, mouse, session_pair: tuple, subset="all"):
        """
        Plot ensemble activations of two matched ensembles across time for two sessions.
        Can also plot two matched ensembles across session halves.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_pair: tuple
            Two sessions you want to match. If the two sessions are the same, the function matches the split-session
            ensembles instead.

        subset: str or array-like
            If 'all', plots all ensembles. Otherwise, plots the ensemble numbers from the first value in the session_pair tuple.


        """
        split_session = True if len(np.unique(session_pair)) == 1 else False
        registered_ensembles = self.match_ensembles(mouse, session_pair)

        if split_session:
            session = self.data[mouse][session_pair[0]]
            spike_times = session.imaging["spike_times"]
            n_frames = len(session.imaging["frames"])

            halfway = np.round(n_frames / 2)

            # Split the spike times down the midpoint of the session.
            registered_spike_times = [
                [spikes[spikes < halfway] for spikes in spike_times]
            ]
            registered_spike_times.append(
                [spikes[spikes >= halfway] for spikes in spike_times]
            )

            frames = [np.arange(halfway), np.arange(halfway, n_frames)]
        else:
            registered_spike_times = self.rearrange_neurons(
                mouse, session_pair, "spike_times", detected="everyday"
            )
            frames = None

        if subset == "all":
            subset = range(registered_ensembles["matched_patterns"].shape[1])

        for i, (
            s1_activation,
            s2_activation,
            s1_pattern,
            s2_pattern,
            poor_match,
            similarity,
        ) in enumerate(
            zip(
                registered_ensembles["matched_activations"][0][subset],
                registered_ensembles["matched_activations"][1][subset],
                registered_ensembles["matched_patterns"][0][subset],
                registered_ensembles["matched_patterns"][1][subset],
                registered_ensembles["poor_matches"][subset],
                registered_ensembles["best_similarities"][subset],
            )
        ):
            fig = self.plot_matched_ensemble(
                (s1_activation, s2_activation),
                (s1_pattern, s2_pattern),
                registered_spike_times,
                similarity,
                session_pair,
                poor_match=poor_match,
                split_session=split_session,
                frames=frames,
            )

        return fig

    def spiralplot_matched_ensembles(
        self, mouse, session_pair: tuple, thresh=1, subset="all"
    ):
        # Match assemblies.
        registered_ensembles = self.match_ensembles(mouse, session_pair)

        if subset == "all":
            subset = range((len(registered_ensembles["matches"])))

        # Get timestamps and linearized position.
        t = [
            self.data[mouse][session].behavior.data["df"]["t"]
            for session in session_pair
        ]
        linearized_position = [
            self.data[mouse][session].behavior.data["df"]["lin_position"]
            for session in session_pair
        ]

        # For each assembly in session 1 and its corresponding match in session 2, get their activation profiles.
        for s1_assembly, (s2_assembly, poor_match) in enumerate(
            zip(registered_ensembles["matches"], registered_ensembles["poor_matches"])
        ):
            if s1_assembly in subset:
                activations = [
                    self.data[mouse][session_type].assemblies["activations"][assembly]
                    for session_type, assembly in zip(
                        session_pair, [s1_assembly, s2_assembly]
                    )
                ]

                # Make a figure with 2 subplots, one for each session.
                fig, axs = plt.subplots(
                    2, 1, subplot_kw=dict(polar=True), figsize=(6.4, 10)
                )
                plot_legend = True
                for ax, activation, t_, lin_pos, assembly_number, session_type in zip(
                    axs,
                    activations,
                    t,
                    linearized_position,
                    [s1_assembly, s2_assembly],
                    session_pair,
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
                        plot_legend=plot_legend,
                    )
                    plot_legend = False
                    title_color = "r" if poor_match else "k"
                    ax.set_title(
                        f"Ensemble #{assembly_number} in {session_type.replace('Goals','Training')}",
                        color=title_color,
                    )

                    behavior_data = self.data[mouse][session_type].behavior.data
                    port_locations = np.asarray(behavior_data["lin_ports"])[
                        behavior_data["rewarded_ports"]
                    ]
                    [ax.axvline(port, color="y") for port in port_locations]

                    # if session_type == "Reversal":
                    #     behavior_data = self.data[mouse]["Goals4"].behavior.data
                    #     # NEED TO FINISH PLOTTING REWARD SITES

                if self.save_configs["save_figs"]:
                    self.save_fig(
                        fig, f"{mouse}_ensemble{s1_assembly}_from_{session_pair[0]}", 3
                    )

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
        for age in ages:
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
                color_boxes(boxes, [color, color])

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
        for age in ages:
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

        for age, ax, color in zip(ages, axs, age_colors):
            boxes = ax.boxplot(percent_matches[age].values(), patch_artist=True)
            color_boxes(boxes, color)

            if age == "aged":
                ax.set_yticks([])
            else:
                ax.set_ylabel("Percent matched ensembles")
            ax.set_xticklabels([session_pair1, session_pair2], rotation=45)
            ax.set_title(age)
        fig.tight_layout()

        return percent_matches

    # def matched_ensemble_fading_prop(self, session_pair=['Reversal', 'Goals4']):
    #     registered_ensembles, fading_that_got_matched, \
    #     p_of_matched_that_is_fading, rising_that_did_not_match, \
    #     p_of_unmatched_that_is_rising = dict(), dict(), dict(), dict(), dict()
    #     ensemble_trends, ensemble_counts = self.plot_assembly_trends(session_pair, show_plot=False)
    #
    #     for age in ages:
    #         for mouse in self.meta['grouped_mice'][age]:
    #             registered_ensembles[mouse] = self.match_ensembles(mouse, session_pair)
    #
    #             matched = np.where(~registered_ensembles[mouse]['poor_matches'])[0]
    #             unmatched = np.where(registered_ensembles[mouse]['poor_matches'])[0]
    #
    #             fading = ensemble_trends.sel(mouse=mouse, trend='decreasing', session='Reversal').values.tolist()
    #             rising = ensemble_trends.sel(mouse=mouse, trend='increasing', session='Reversal').values.tolist()
    #
    #             fading_that_got_matched[mouse] = np.sum([f in matched for f in fading])/len(fading)
    #             p_of_matched_that_is_fading[mouse] = np.sum([m in fading for m in matched])/len(matched)
    #             rising_that_did_not_match[mouse] = np.sum([r in unmatched for r in rising])/len(rising)
    #             p_of_unmatched_that_is_rising[mouse] = np.sum([u in rising for u in unmatched])/len(unmatched)
    #
    #     return fading_that_got_matched, p_of_matched_that_is_fading, rising_that_did_not_match, p_of_unmatched_that_is_rising

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
                color_boxes(boxes, color)

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
            for age in ages
        }
        for age in ages:
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

            color_boxes(boxes, age_colors)

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

    def get_spatial_ensembles(self, mouse, session_type, alpha=0.05, method="sidak"):
        if method is None:
            is_spatial = (
                self.data[mouse][session_type]
                .assemblies["fields"]
                .data["spatial_info_pvals"]
                < alpha
            )
        else:
            is_spatial = multipletests(
                self.data[mouse][session_type]
                .assemblies["fields"]
                .data["spatial_info_pvals"],
                alpha=alpha,
                method=method,
            )[0]

        return is_spatial

    def find_proportion_spatial_ensembles(self, pval_threshold=0.05, method="sidak"):
        p = dict()
        for session in self.meta["session_types"]:
            p[session] = dict()

            for age in ages:
                p[session][age] = []

                for mouse in self.meta["grouped_mice"][age]:
                    is_spatial = self.get_spatial_ensembles(
                        mouse, session, pval_threshold, method=method
                    )

                    p[session][age].append(sum(is_spatial) / len(is_spatial))

        fig, axs = plt.subplots(1, len(self.meta["session_types"]), sharey=True)
        fig.subplots_adjust(wspace=0)

        for i, (ax, session) in enumerate(zip(axs, self.meta["session_types"])):
            self.scatter_box(p[session], ax=ax)

            ax.set_xticks([])
            ax.set_title(session.replace("Goals", "Training"), fontsize=14)

        axs[0].set_ylabel("Proportion of\nspatial ensembles")

        self.set_age_legend(fig)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0)

        return p

    def find_proportion_of_fading_ensembles_that_are_spatial(
        self, session_type, alpha=0.05, ages_to_plot="young", method="sidak"
    ):
        reversal = True if session_type == "Reversal" else False
        p_spatial = pd.DataFrame()

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        for age in ages_to_plot:
            for mouse in self.meta["grouped_mice"][age]:
                fading = self.find_activity_trends(mouse, "Reversal")[0]["decreasing"]

                if not reversal:
                    registered_ensembles = self.match_ensembles(
                        mouse, ("Reversal", session_type)
                    )
                    poor_matches = np.where(registered_ensembles["poor_matches"])[0]
                    fading = [f for f in fading if f not in poor_matches]
                    fading = registered_ensembles["matches"][fading]

                is_spatial = self.get_spatial_ensembles(
                    mouse, session_type, alpha=alpha, method=method
                )

                p = np.nanmean(is_spatial[fading])
                chance = [
                    np.nanmean(random.sample(list(is_spatial), len(fading)))
                    for i in range(1000)
                ]

                if len(fading):
                    pval = np.sum(p < np.asarray(chance)) / len(chance)
                else:
                    pval = np.nan

                p_spatial = pd.concat(
                    (
                        p_spatial,
                        pd.DataFrame(
                            {
                                "mice": mouse,
                                "age": age,
                                "session": session_type,
                                "p_spatial": [p],
                                "chance": [chance],
                                "pvalue": [pval],
                            }
                        ),
                    )
                )

        p_spatial = p_spatial.dropna()

        fig, axs = plt.subplots(1, n_ages_to_plot)
        axs = [axs] if n_ages_to_plot == 1 else axs
        for age, color, ax in zip(ages_to_plot, plot_colors, axs):
            idx = p_spatial["age"] == age
            boxes = ax.boxplot(
                p_spatial.loc[idx, "chance"],
                showfliers=False,
                zorder=0,
                widths=0.75,
                patch_artist=True,
            )
            color_boxes(boxes, "w")
            ax.plot(
                np.arange(1, len(idx) + 1),
                p_spatial.loc[idx, "p_spatial"],
                "_",
                markersize=18,
                color=color,
                markeredgewidth=5,
                zorder=1,
            )
            ax.set_ylabel(
                f"P(spatial during {session_type.replace('Goals', 'Training')} "
                f"|\nfading during Reversal)",
                fontsize=22,
            )
            ax.set_xticklabels(np.unique(p_spatial.loc[idx, "mice"]), rotation=45)
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        return p_spatial

    def SI_comparison_of_fading_ensembles(
        self,
        session_type,
        ages_to_plot="young",
        alpha=0.05,
        method="sidak",
        spatial_only=True,
    ):
        SIs_df = pd.DataFrame()
        reversal = True if session_type == "Reversal" else False

        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        for age in ages_to_plot:
            for mouse in self.meta["grouped_mice"][age]:
                fading = self.find_activity_trends(mouse, "Reversal")[0]["decreasing"]

                if not reversal:
                    registered_ensembles = self.match_ensembles(
                        mouse, ("Reversal", session_type)
                    )
                    poor_matches = np.where(registered_ensembles["poor_matches"])[0]
                    fading = [f for f in fading if f not in poor_matches]
                    fading = registered_ensembles["matches"][fading]

                is_spatial = np.where(
                    self.get_spatial_ensembles(
                        mouse, session_type, alpha=alpha, method=method
                    )
                )[0]
                fading = [f for f in fading if f in is_spatial]

                SIs = (
                    self.data[mouse][session_type]
                    .assemblies["fields"]
                    .data["spatial_info_z"]
                )
                SI_fading = SIs[fading]
                mean_SIs_fading = np.nanmean(SI_fading)
                if spatial_only:
                    SIs_to_sample = SIs[is_spatial]
                else:
                    SIs_to_sample = SIs

                chance = [
                    np.nanmean(random.sample(list(SIs_to_sample), len(fading)))
                    for i in range(5000)
                ]
                if len(fading):
                    pval = np.sum(mean_SIs_fading < np.asarray(chance)) / len(chance)
                else:
                    pval = np.nan

                SIs_df = pd.concat(
                    (
                        SIs_df,
                        pd.DataFrame(
                            {
                                "mice": mouse,
                                "age": age,
                                "session": session_type,
                                "SIs": [SI_fading],
                                "chance": [chance],
                                "pvalue": [pval],
                            }
                        ),
                    )
                )
        SIs_df = SIs_df.dropna()
        fig, axs = plt.subplots(1, n_ages_to_plot, sharey=True)
        axs = [axs] if n_ages_to_plot == 1 else axs
        for age, color, ax in zip(ages_to_plot, plot_colors, axs):
            idx = SIs_df["age"] == age
            boxes = ax.boxplot(
                SIs_df.loc[idx, "chance"],
                showfliers=False,
                zorder=0,
                widths=0.75,
                patch_artist=True,
            )
            color_boxes(boxes, "w")
            ax.plot(
                np.arange(1, len(idx) + 1),
                [np.nanmean(si) for si in SIs_df.loc[idx, "SIs"]],
                "_",
                markersize=18,
                color=color,
                markeredgewidth=5,
                zorder=1,
            )
            chance = np.nanmean([np.nanmean(i) for i in SIs_df.loc[idx, "chance"]])
            ax.axhline(y=chance, color="r")
            ax.text(
                1.1, chance + 0.5, "Chance", horizontalalignment="center", color="r"
            )

            ax.set_ylabel(
                f"Spatial info. of Reversal fading\nensembles during "
                f"{session_type.replace('Goals', 'Training')} (z)",
                fontsize=18,
            )
            ax.set_xticklabels(np.unique(SIs_df.loc[idx, "mice"]), rotation=45)
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]

        fig.tight_layout()
        return SIs_df

    def fading_also_fading_on_other_sessions(
        self, session_pair=("Reversal", "Goals4"), ages_to_plot="young"
    ):
        """
        Of the fading ensembles from session_pair[0], what proportion of those are fading also in
        session_pair[1]?

        :param session_pair:
        :param ages_to_plot:
        :return:
        """
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        p_also_fading = dict()
        chance = dict()
        for age in ages_to_plot:
            for mouse in self.meta["grouped_mice"][age]:
                trends = {
                    session: self.find_activity_trends(mouse, session)[0]
                    for session in session_pair
                }
                n_fading = len(trends[session_pair[0]]["decreasing"])

                registered_ensembles = self.match_ensembles(mouse, session_pair)
                poor_matches = np.where(registered_ensembles["poor_matches"])[0]
                fading = trends[session_pair[0]]["decreasing"]
                fading = [f for f in fading if f not in poor_matches]
                fading = registered_ensembles["matches"][fading]

                p_also_fading[mouse] = (
                    np.sum(
                        [
                            1
                            for f in fading
                            if f in trends[session_pair[1]]["decreasing"]
                        ]
                    )
                    / n_fading
                )

                chance[mouse] = []
                n_ensembles = (
                    self.data[mouse][session_pair[1]]
                    .assemblies["significance"]
                    .nassemblies
                )
                for i in np.arange(1000):
                    randsamp = random.sample(range(n_ensembles), n_fading)
                    chance[mouse].append(
                        np.sum([1 for f in fading if f in randsamp]) / n_fading
                    )

        return p_also_fading, chance

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
            raise NotImplementedError(
                f"Invalid value for reference_session: {reference_session}"
            )

        # Register the ensembles.
        registered_ensembles = self.match_ensembles(mouse, session_pair)

        ensemble_field_data = [
            self.make_ensemble_fields(
                mouse,
                session,
                spatial_bin_size_radians=spatial_bin_size_radians,
                running_only=running_only,
                std_thresh=std_thresh,
                get_zSI=False,
            )
            for i, session in enumerate(session_pair)
        ]

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
            ensemble_fields[0], ensemble_fields[1] = (
                ensemble_fields[1],
                ensemble_fields[0],
            )
            ensemble_field_data[0], ensemble_field_data[1] = (
                ensemble_field_data[1],
                ensemble_field_data[1],
            )

        return ensemble_fields, ensemble_field_data

    def correlate_ensemble_fields(
        self,
        mouse,
        session_types,
        spatial_bin_size_radians=0.05,
        subset=None,
        n_shuffles=None,
    ):
        ensemble_fields = self.map_ensemble_fields(
            mouse, session_types, spatial_bin_size_radians=spatial_bin_size_radians
        )[0]

        if subset is None:
            subset = range(len(ensemble_fields[0]))

        rhos = [
            spearmanr(ensemble_day1, ensemble_day2)[0]
            for ensemble_day1, ensemble_day2 in zip(
                ensemble_fields[0][subset], ensemble_fields[1][subset]
            )
        ]

        if n_shuffles is not None:
            shuffle_rhos = []
            ensembles_1 = ensemble_fields[0][subset]

            shuffles = tqdm(range(n_shuffles))
            for i in shuffles:
                ensembles_2 = np.random.permutation(ensemble_fields[1][subset])

                shuffle_rhos.append(
                    [spearmanr(a, b)[0] for a, b in zip(ensembles_1, ensembles_2)]
                )
        else:
            shuffle_rhos = None

        return np.asarray(rhos), shuffle_rhos

    def fading_ensemble_spatial_corr(self, session_type="Goals4", ages_to_plot="young"):
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )

        rho_df = pd.DataFrame()
        for age in ages_to_plot:
            for mouse in self.meta["grouped_mice"][age]:
                trends = self.find_activity_trends(mouse, "Reversal")[0]
                rhos, shuffle_rhos = self.correlate_ensemble_fields(
                    mouse,
                    ("Reversal", session_type),
                    subset=trends["decreasing"],
                    n_shuffles=100,
                )

                rho_dict = {
                    "age": age,
                    "mouse": mouse,
                    "mean_rho": [np.nanmean(rhos)],
                    "mean_shuffle": [np.nanmean(np.hstack(shuffle_rhos))],
                }

                rho_df = pd.concat((rho_df, pd.DataFrame(rho_dict)))

        fig, axs = plt.subplots(
            1, n_ages_to_plot, sharey=True, figsize=(3.5 * n_ages_to_plot, 4.8)
        )
        axs = [axs] if n_ages_to_plot else axs
        for age, color, ax in zip(ages_to_plot, plot_colors, axs):
            real = rho_df.loc[rho_df["age"] == age, "mean_rho"]
            chance = rho_df.loc[rho_df["age"] == age, "mean_shuffle"]
            boxes = ax.boxplot(
                [real[~np.isnan(real)], chance[~np.isnan(chance)]],
                patch_artist=True,
                zorder=0,
                widths=0.75,
                showfliers=False,
            )
            color_boxes(boxes, color)

            ax.plot(
                [jitter_x(np.ones_like(real)), jitter_x(np.ones_like(chance) * 2)],
                [real, chance],
                "o-",
                color="k",
                markerfacecolor=color,
                zorder=1,
                markersize=10,
            )

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Real", "Shuffled"], rotation=45)
            ax.set_ylabel("Fading ensemble" "\n" r"spatial correlation [$\rho$]")
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]

        fig.suptitle(f'Reversal x {session_type.replace("Goals", "Training")}')
        fig.tight_layout()

        stats = ttest_rel(rho_df["mean_rho"].dropna(), rho_df["mean_shuffle"].dropna())
        msg = f"t={np.round(stats.statistic, 3)}, p={stats.pvalue}"
        print(msg)

        return rho_df

    def plot_ensemble_field_correlations(self, session_pair: tuple, show_plot=True):
        ensemble_field_rhos = dict()
        for age in ages:
            ensemble_field_rhos[age] = []

            for mouse in self.meta["grouped_mice"][age]:
                rhos = self.correlate_ensemble_fields(mouse, session_pair)[0]
                ensemble_field_rhos[age].append(rhos[~np.isnan(rhos)])

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, color, ax in zip(ages, age_colors, axs):
                boxes = ax.boxplot(ensemble_field_rhos[age], patch_artist=True)

                color_boxes(boxes, color)

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
                    color_boxes(boxes, color)

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
                np.nanmedian(self.correlate_ensemble_fields(mouse, session_pair)[0])
                for mouse in self.meta["grouped_mice"][age]
            ]
            for age in ages
        }

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 5))
        boxes = ax.boxplot([rhos[age] for age in ages], widths=0.75, patch_artist=True)
        color_boxes(boxes, age_colors)
        ax.set_xticklabels(ages)
        ax.set_title(session_pair)
        ax.set_ylabel("Ensemble field correlation [Spearman rho]")
        plt.tight_layout()

        return rhos

    def xcorr_ensemble_cells(
        self,
        mouse,
        session_type,
        ensemble_number,
        data_type="S",
        n_splits=6,
        show_plot=True,
        mat_to_cluster=0,
        # n_clusters=4,
    ):
        """
        Do pairwise correlations between ensemble cells.

        """
        session = self.data[mouse][session_type]
        pattern = session.assemblies["patterns"][ensemble_number]

        # Find ensemble members
        members = find_members(pattern, filter_method="sd", thresh=2)[1]

        # Split the activity into n equal parts.
        traces = np.array_split(
            session.imaging[data_type][members].astype(float), n_splits, axis=1
        )
        imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
        traces = [imp.fit_transform(t) for t in traces]

        # Do correlation, nan the diagonal, find min/max.
        n_neurons = len(members)

        corr_mats, pval_mats = [], []
        for t in traces:
            R = nan_array((n_neurons, n_neurons))
            pval_mat = nan_array((n_neurons, n_neurons))
            for combination in product(range(n_neurons), repeat=2):
                if combination[0] != combination[1]:
                    x = t[combination[0]]
                    y = t[combination[1]]

                    r, p = spearmanr(x, y)
                    if not np.isfinite(r):
                        r = 0
                        p = 1
                    R[combination[0], combination[1]] = r
                    pval_mat[combination[0], combination[1]] = p
            corr_mats.append(R)
            pval_mats.append(pval_mat)

        # kmeans = KMeans(n_clusters=3)
        # labels = kmeans.fit_predict(corr_mats[mat_to_cluster])
        # idx = np.argsort(labels)
        labels, idx, linkage = cluster_corr(corr_mats[mat_to_cluster])
        # cluster_labels = cut_tree(linkage, n_clusters=n_clusters).flatten()
        idx = np.argsort(labels)
        [np.fill_diagonal(c, np.nan) for c in corr_mats]

        if show_plot:
            self.plot_cell_xcorr_matrices(corr_mats, idx, axs=None)

        data = {
            "neurons": members,
            "correlations": corr_mats,
            "pvals": pval_mats,
            "traces": traces,
            "labels": labels,
            "linkage": linkage,
            # "cluster_labels": cluster_labels,
        }

        return data

    def plot_cell_xcorr_matrices(self, corr_mats, idx, axs=None, plot_cbar=False):
        n_splits = len(corr_mats)
        n_neurons = corr_mats[0].shape[0]
        if axs is None:
            fig, axs = plt.subplots(
                1, n_splits, figsize=(2 * n_splits, 2), sharex=True, sharey=True
            )
        else:
            fig = axs[0].figure

        vmax = np.nanmax([np.nanmax(c) for c in corr_mats])
        vmin = np.nanmin([np.nanmin(c) for c in corr_mats])

        # Center color bar on 0.
        midpoint = 1 - vmax / (vmax + abs(vmin))
        cmap = shiftedColorMap(matplotlib.cm.bwr, start=vmin, midpoint=midpoint)

        interval_min = int(1800 / n_splits / 60)
        time_bins = np.append(np.arange(0, 30, interval_min), 30)
        titles = [
            f"{start}-{stop} min" for start, stop in zip(time_bins[:-1], time_bins[1:])
        ]
        for ax, mat, title in zip(axs.flatten(), corr_mats, titles):
            im = ax.imshow(
                mat[idx, :][:, idx], vmin=vmin, vmax=vmax, cmap=cmap, origin="lower"
            )
            ax.set_xticks([0, n_neurons])
            ax.set_yticks([0, n_neurons])
            ax.set_title(title, fontsize=20)

        if plot_cbar:
            cbar_fig, _ = plt.subplots()
            cbar_fig.colorbar(im, shrink=0.5)
        else:
            cbar_fig = None
        # fig.suptitle(f'Ensemble #{ensemble_number}')
        fig.tight_layout()

        return fig, cbar_fig

    def make_graphs(self, data, method="fdr_bh"):
        A = [np.zeros(pvals.shape) for pvals in data["pvals"]]
        for i, (pvals, R) in enumerate(zip(data["pvals"], data["correlations"])):
            np.fill_diagonal(pvals, 1)
            reject, pvals_corr = multipletests(pvals.flatten(), method=method)[:2]
            r, c = np.unravel_index(np.where(reject), pvals.shape)

            A[i][r, c] = R[r, c]

        node_mappings = {i: neuron_id for i, neuron_id in enumerate(data["neurons"])}
        G = [
            nx.relabel_nodes(nx.convert_matrix.from_numpy_matrix(a), node_mappings)
            for a in A
        ]

        return G

    def plot_graph_evolution(
        self,
        mouse,
        session_type,
        ensemble_number,
        n_splits=6,
        method="fdr_bh",
        plot_cbar=False,
    ):
        data = self.xcorr_ensemble_cells(
            mouse, session_type, ensemble_number, n_splits=n_splits, show_plot=False
        )
        G = self.make_graphs(data, method=method)

        fig, axs = plt.subplots(2, n_splits, figsize=(2 * n_splits, 4.5))
        cbar_fig = self.plot_cell_xcorr_matrices(
            data["correlations"],
            idx=np.argsort(data["labels"]),
            axs=axs[0],
            plot_cbar=plot_cbar,
        )[1]
        for ax, g in zip(axs[1], G):
            pos = nx.drawing.layout.circular_layout(g)
            cc = average_clustering(g)
            nx.draw(g, pos=pos, ax=ax, node_size=10, width=0.05)
            ax.axis("square")
            ax.set_title(f"CC = {cc}", fontsize=18)
        axs[0, 0].set_ylabel("Ensemble\nmember correlations", fontsize=18)
        axs[1, 0].set_axis_on()
        [spine.set_visible(False) for spine in axs[1, 0].spines.values()]
        axs[1, 0].set_ylabel("Network\nconnections", fontsize=18)

        fig.tight_layout()

        return fig, cbar_fig

    def plot_graphs_deg_labels(
        self,
        mouse,
        session_type,
        ensemble_number,
        n_splits=6,
        method="fdr_bh",
        quartile_on=-1,
        axs=None,
    ):
        data = self.xcorr_ensemble_cells(
            mouse, session_type, ensemble_number, n_splits=n_splits, show_plot=False
        )
        G = self.make_graphs(data, method=method)

        degrees = [d for n, d in G[quartile_on].degree()]
        percentiles = np.percentile(degrees, [0, 25, 50, 75, 100])
        percentiles[0] = percentiles[0] - 1
        quartile_memberships = []
        for i, (lower, upper) in enumerate(zip(percentiles[:-1], percentiles[1:])):
            quartile_memberships.append(
                np.where(np.logical_and(degrees > lower, degrees <= upper))[0]
            )

        node_colors = []
        for i in range(len(degrees)):
            if i in quartile_memberships[0]:
                node_colors.append("orange")
            elif i in quartile_memberships[-1]:
                node_colors.append("limegreen")
            else:
                node_colors.append("k")

        if axs is None:
            fig, axs = plt.subplots(1, n_splits)
        else:
            fig = axs[0].figure

        for ax, g in zip(axs, G):
            pos = nx.drawing.layout.spring_layout(g)
            nx.draw(g, node_color=node_colors, pos=pos, ax=ax, node_size=10, width=0.2)
            ax.axis("square")

        return fig

    def plot_degree_distribution(self, mouse, session, ensemble,
                                 n_splits=6, method="fdr_bh", colors=["orange", "k", "limegreen"]):
        G = self.make_graphs(self.xcorr_ensemble_cells(mouse, session, ensemble, n_splits=n_splits,
                                                       show_plot=False), method=method)
        degrees = np.asarray([d for n,d in G[-1].degree()])
        percentiles = np.percentile(degrees, [0, 25, 50, 75, 100])
        percentiles[0] = percentiles[0] - 1
        x = [degrees[np.logical_and(degrees > percentiles[0], degrees <= percentiles[1])],
             degrees[np.logical_and(degrees > percentiles[1], degrees <= percentiles[-2])],
             degrees[np.logical_and(degrees > percentiles[-2], degrees <= percentiles[-1])]]

        fig, ax = plt.subplots()
        ax.hist(x, color=colors, rwidth=200)
        ax.set_ylabel("Number of neurons")
        ax.set_xlabel("Degrees")
        [ax.spines[side].set_visible(False) for side in ['top', 'right']]
        fig.tight_layout()

        return fig

    def xcorr_fading_ensembles(
        self, mouse, session_type, n_splits=6, trend="decreasing"
    ):
        ensemble_trends = self.find_activity_trends(mouse, session_type)[0]

        if not ensemble_trends[trend]:
            return None

        xcorrs = dict()
        for ensemble in ensemble_trends[trend]:
            xcorrs[ensemble] = self.xcorr_ensemble_cells(
                mouse, session_type, ensemble, n_splits=n_splits, show_plot=False
            )

        return xcorrs

    def degree_start_end_session(
        self,
        mouse,
        session_type,
        n_splits=6,
        method="fdr_bh",
        quartiles_to_plot=None,
        trend="decreasing",
        axs=None,
    ):
        degree_df = self.find_degrees(
            mouse, session_type, n_splits=n_splits, method=method, trend=trend
        )
        last_time_bin = n_splits - 1
        degree_df = self.categorize_hub_dropped_neurons(
            degree_df, time_bin=last_time_bin
        )

        if degree_df.empty:
            return None, None

        age = "aged" if mouse in self.meta["grouped_mice"]["aged"] else "young"
        color = age_colors[ages.index(age)]

        if quartiles_to_plot:
            titles = ["1st quartile", "2nd quartile", "3rd quartile", "4th quartile"]
            titles = [titles[i - 1] for i in quartiles_to_plot]
            n_percentiles = len(quartiles_to_plot)

            if axs is None:
                fig, axs = plt.subplots(
                    1, n_percentiles, sharey=True, figsize=(2.5 * n_percentiles, 6)
                )
                axs[0].set_ylabel("Mean degree per fading ensemble")
            else:
                fig = axs[0].figure
            for title, quartile, ax in zip(titles, quartiles_to_plot, axs):
                m = (
                    degree_df.groupby(["quartile", "time_bin", "ensemble_id"])
                    .mean()
                    .loc[quartile, "degree"]
                )
                data = [m.loc[i] for i in [0, last_time_bin]]
                boxes = ax.boxplot(
                    [y for y in data], patch_artist=True, widths=0.75, zorder=0
                )
                ax.plot([1, 2], [y for y in data], color=color, zorder=1, alpha=0.5)
                color_boxes(boxes, "w")
                ax.set_title(title)
                ax.set_xticklabels(
                    ["Beginning\nof session", "End of\nsession"], rotation=45
                )
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]

            fig.tight_layout()
            fig.subplots_adjust(wspace=0.3)
        else:
            fig = None

        return degree_df, fig

    def find_degrees(
        self,
        mouse,
        session_type,
        n_splits=6,
        method="fdr_bh",
        trend="decreasing",
        subgraphs=None,
    ):
        xcorrs = self.xcorr_fading_ensembles(
            mouse, session_type, n_splits=n_splits, trend=trend
        )
        if xcorrs is None:
            return pd.DataFrame()

        G_ensembles = [
            self.make_graphs(ensemble, method=method) for ensemble in xcorrs.values()
        ]

        if subgraphs is not None:
            G_subgraphed = []
            for G_ensemble, subgraph in zip(G_ensembles, subgraphs):
                G_subgraphed.append([G.subgraph(subgraph) for G in G_ensemble])

        else:
            G_subgraphed = G_ensembles

        degree_df = pd.DataFrame()
        for i, (G_seq, ensemble_id) in enumerate(zip(G_subgraphed, xcorrs.keys())):
            for time_bin, G in enumerate(G_seq):
                degree = {n: d for n, d in G.degree()}

                degree_df = pd.concat(
                    (
                        degree_df,
                        pd.DataFrame(
                            {
                                "mouse": mouse,
                                "neuron_id": degree.keys(),
                                "ensemble_id": ensemble_id,
                                "time_bin": time_bin,
                                "degree": degree.values(),
                            }
                        ),
                    ),
                    ignore_index=True,
                )

        return degree_df.reset_index(drop=True)

    def categorize_hub_dropped_neurons(self, session_degree_df, time_bin=5):
        if "time_bin" not in session_degree_df:
            return session_degree_df

        time_bin_idx = session_degree_df["time_bin"] == time_bin
        degrees_this_bin = session_degree_df.loc[time_bin_idx]
        for ensemble_id in degrees_this_bin.ensemble_id.unique():
            ensemble_idx = session_degree_df["ensemble_id"] == ensemble_id
            idx = np.logical_and(time_bin_idx, ensemble_idx)
            ensemble_degrees = degrees_this_bin.loc[idx]

            quartiles = np.percentile(ensemble_degrees["degree"], [0, 25, 50, 75, 100])
            quartiles[0] = quartiles[0] - 1

            quartile_bins = np.digitize(
                ensemble_degrees["degree"], quartiles, right=True
            )
            quartile_map = {
                neuron_id: percentile
                for neuron_id, percentile in zip(
                    ensemble_degrees["neuron_id"], quartile_bins
                )
            }

            session_degree_df.loc[ensemble_idx, "quartile"] = session_degree_df.loc[
                ensemble_idx, "neuron_id"
            ].map(quartile_map)

        return session_degree_df

    def make_degree_df_with_quartiles(
        self, mouse, session_type, trend="decreasing", time_bin=5
    ):
        degree_df = self.find_degrees(mouse, session_type, trend=trend)
        degree_df = self.categorize_hub_dropped_neurons(degree_df, time_bin=time_bin)

        return degree_df

    def contour_degree_beginning_and_end(
        self, age, session_type="Reversal", trend="decreasing"
    ):
        mice = self.meta["grouped_mice"][age]

        Degrees = pd.DataFrame()
        for mouse in mice:
            degree_df = self.make_degree_df_with_quartiles(
                mouse, session_type, trend=trend, time_bin=5
            )
            Degrees = pd.concat((Degrees, degree_df), ignore_index=True)
        mice = Degrees.mouse.unique()

        fig, axs = plt.subplots(
            1, len(mice), sharey=True, sharex=True, figsize=(len(mice) * 3, 6)
        )
        for ax, mouse in zip(axs, mice):
            mouse_df = Degrees.loc[Degrees["mouse"] == mouse]
            mouse_df.drop_duplicates(["neuron_id", "time_bin"], inplace=True)

            time_bin0 = mouse_df.loc[
                np.logical_and(
                    mouse_df["time_bin"] == 0, mouse_df["quartile"].isin([1, 4])
                ),
                ["neuron_id", "degree", "quartile"],
            ]

            time_bin5 = mouse_df.loc[
                np.logical_and(
                    mouse_df["time_bin"] == 5, mouse_df["quartile"].isin([1, 4])
                ),
                ["neuron_id", "degree", "quartile"],
            ]

            merged = time_bin0.merge(
                time_bin5, how="outer", left_on="neuron_id", right_on="neuron_id"
            )
            sns.kdeplot(
                data=merged,
                x="degree_x",
                y="degree_y",
                hue="quartile_x",
                ax=ax,
                palette={1: "orange", 4: "cyan"},
                legend=False,
            )

            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_xlabel("Degree @ begin.\nof session")
            ax.set_ylabel("Degree @ end\nof session")
            ax.set_title(f"Mouse {mouse[0]}")

        [ax.set_aspect("equal") for ax in axs]
        for ax in axs:
            ax.set_xlim([0, Degrees.degree.max()])
            ax.set_ylim([0, Degrees.degree.max()])
            plot_xy_line(ax)
        fig.tight_layout()

        return Degrees, fig

    def degree_comparison_all_mice(
        self, age, average_within_ensemble=True, trend="decreasing", plot_agg=False
    ):
        """
        Compute degrees for all neurons based on the graph from the last 5 min of the
        session. The neurons with the lowest quartile of degrees are called "dropped",
        and the neurons with the highest quartile of degrees are called "hubs". At the
        first 5 min of the session, do the dropped neurons already have lower degrees
        than hub neurons?

        :parameters
        age: str, 'young' or 'aged'

        average_within_ensembles: bool
            If True, take the average degree of dropped neurons within an ensemble.
        """
        mice = self.meta["grouped_mice"][age]
        color = age_colors[ages.index(age)]
        ylabel = {
            True: "Average degree",
            False: "Degree",
        }
        Degrees = pd.DataFrame()
        for mouse in mice:
            degree_df = self.make_degree_df_with_quartiles(
                mouse, "Reversal", trend=trend, time_bin=5
            )
            Degrees = pd.concat((Degrees, degree_df), ignore_index=True)

        if plot_agg:
            fig, ax = plt.subplots(figsize=(4, 4.8))
            boxes = ax.boxplot(
                [
                    Degrees.loc[
                        np.logical_and(
                            Degrees["time_bin"] == 0, Degrees["quartile"] == i
                        ),
                        "degree",
                    ]
                    for i in [4, 1]
                ],
                widths=0.75,
                patch_artist=True,
            )
            color_boxes(boxes, color)

            ax.set_xticklabels(
                ["Hub\nneurons", "Dropped\nneurons"], rotation=45, fontsize=14
            )
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_ylabel("Degree at beginning\nof session", fontsize=22)
            t_tests = None
        else:
            n_mice = len(Degrees.mouse.unique())
            fig, axs = plt.subplots(1, n_mice, sharey=True, figsize=(2 * n_mice, 4.8))
            t_tests = dict()
            for mouse, ax in zip(mice, axs):
                mouse_degrees = Degrees.loc[Degrees["mouse"] == mouse]
                if average_within_ensemble:
                    hub_neurons = mouse_degrees.loc[mouse_degrees["quartile"] == 4]
                    stable = hub_neurons.groupby(["time_bin", "ensemble_id"]).mean()[
                        "degree"
                    ][0]

                    dropped_neurons = mouse_degrees.loc[mouse_degrees["quartile"] == 1]
                    disconnected = dropped_neurons.groupby(
                        ["time_bin", "ensemble_id"]
                    ).mean()["degree"][0]
                    showfliers = False
                else:
                    first_time_bin = mouse_degrees["time_bin"] == 0
                    stable = mouse_degrees.loc[
                        np.logical_and(first_time_bin, mouse_degrees["quartile"] == 4),
                        "degree",
                    ]
                    disconnected = mouse_degrees.loc[
                        np.logical_and(first_time_bin, mouse_degrees["quartile"] == 1),
                        "degree",
                    ]
                    showfliers = True

                boxes = ax.boxplot(
                    [stable, disconnected],
                    widths=0.75,
                    patch_artist=True,
                    zorder=0,
                    showfliers=showfliers,
                )

                neuron_colors = cycle(["limegreen", "orange"])
                if average_within_ensemble:
                    x_ = []
                    for i, y in zip([1, 2], [stable, disconnected]):
                        x = jitter_x(np.ones_like(y), 0.05) * i
                        x_.append(x)
                        ax.scatter(x, y, color=next(neuron_colors), edgecolors="k")
                    ax.plot(x_, [stable, disconnected], color="k")
                color_boxes(boxes, ["limegreen", "orange"])

                ax.set_xticklabels(
                    ["Hub\nneurons", "Dropped\nneurons"], rotation=45, fontsize=14
                )
                ax.set_title(f"Mouse {mouse[0]}")
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]

                if average_within_ensemble:
                    t_result = ttest_rel(stable, disconnected)
                else:
                    t_result = ttest_ind(stable, disconnected)
                t_tests[mouse] = t_result

            axs[0].set_ylabel(
                ylabel[average_within_ensemble] + " at\nbeginning of session",
                fontsize=22,
            )

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.3)

        return Degrees, fig, t_tests

    def delta_degree_comparison(self, age="young", plot_subgraph=False):
        Degrees = dict()
        trends = ["decreasing", "no trend"]
        for trend in ["decreasing", "no trend"]:
            degrees = pd.DataFrame()
            for mouse in self.meta["grouped_mice"][age]:
                degree_df = self.make_degree_df_with_quartiles(
                    mouse, "Reversal", trend=trend, time_bin=5
                )
                degrees = pd.concat((degrees, degree_df), ignore_index=True)

            Degrees[trend] = degrees

        # Make subgraph.
        degrees = pd.DataFrame()
        for mouse in self.meta["grouped_mice"][age]:
            mouse_idx = Degrees["decreasing"].mouse == mouse

            subgraph = []
            for ensemble in Degrees["decreasing"].loc[mouse_idx].ensemble_id.unique():
                ensemble_idx = Degrees["decreasing"].ensemble_id == ensemble
                quartile_idx = Degrees["decreasing"].quartile > 1
                non_dropped_neurons = (
                    Degrees["decreasing"]
                    .loc[
                        np.logical_and(
                            np.logical_and(mouse_idx, ensemble_idx), quartile_idx
                        ),
                        "neuron_id",
                    ]
                    .values
                )
                subgraph.append(non_dropped_neurons)

            degree_df = self.find_degrees(
                mouse, "Reversal", trend="decreasing", subgraphs=subgraph
            )
            degrees = pd.concat((degrees, degree_df), ignore_index=True)
        Degrees["no dropped neurons"] = degrees

        color = age_colors[ages.index(age)]
        valid_mice = Degrees["decreasing"].mouse.unique()
        n_mice = len(valid_mice)
        degree_diffs = pd.DataFrame()
        for trend in trends:
            grouped_means = (
                Degrees[trend]
                .groupby(["mouse", "quartile", "time_bin", "ensemble_id"])
                .mean()
            )

            for mouse in valid_mice:
                for quartile in [1, 4]:
                    last_time_bin = grouped_means.loc[mouse, quartile, 5][
                        "degree"
                    ].values
                    first_time_bin = grouped_means.loc[mouse, quartile, 0][
                        "degree"
                    ].values
                    deg_diffs = last_time_bin - first_time_bin

                    degree_diffs = pd.concat(
                        (
                            degree_diffs,
                            pd.DataFrame(
                                {
                                    "mouse": mouse,
                                    "neuron_type": "hub"
                                    if quartile == 4
                                    else "dropped",
                                    "degree_diffs": deg_diffs,
                                    "trend": trend,
                                }
                            ),
                        )
                    )

        for mouse in valid_mice:
            mouse_df = Degrees["decreasing"].loc[Degrees["decreasing"].mouse == mouse]
            deg_diffs = []
            for ensemble in mouse_df.ensemble_id.unique():
                ensemble_df = mouse_df.loc[mouse_df.ensemble_id == ensemble]
                hub_neurons = ensemble_df.loc[
                    ensemble_df["quartile"] == 4, "neuron_id"
                ].unique()

                subgraph_idx = {
                    "mouse": Degrees["no dropped neurons"].mouse == mouse,
                    "ensemble": Degrees["no dropped neurons"].ensemble_id == ensemble,
                    "hub_neurons": Degrees["no dropped neurons"].neuron_id.isin(
                        hub_neurons
                    ),
                    "first_time_bin": Degrees["no dropped neurons"].time_bin == 0,
                    "last_time_bin": Degrees["no dropped neurons"].time_bin == 5,
                }

                deg_t0 = (
                    Degrees["no dropped neurons"]
                    .loc[
                        np.all(
                            [
                                subgraph_idx[key]
                                for key in [
                                    "mouse",
                                    "ensemble",
                                    "hub_neurons",
                                    "first_time_bin",
                                ]
                            ],
                            axis=0,
                        ),
                        "degree",
                    ]
                    .values
                )

                deg_tf = (
                    Degrees["no dropped neurons"]
                    .loc[
                        np.all(
                            [
                                subgraph_idx[key]
                                for key in [
                                    "mouse",
                                    "ensemble",
                                    "hub_neurons",
                                    "last_time_bin",
                                ]
                            ],
                            axis=0,
                        ),
                        "degree",
                    ]
                    .values
                )

                deg_diffs.append(np.nanmean(deg_tf - deg_t0))

            degree_diffs = pd.concat(
                (
                    degree_diffs,
                    pd.DataFrame(
                        {
                            "mouse": mouse,
                            "neuron_type": "hub",
                            "degree_diffs": deg_diffs,
                            "trend": "subgraph",
                        }
                    ),
                )
            )

            # boxes=ax.boxplot(degree_diffs[mouse], patch_artist=True, widths=0.75)
            # color_boxes(boxes, color)
            # ax.set_xticklabels(['Fading\nensembles', 'Non-fading\nensembles'], rotation=45, fontsize=14)
            # [ax.spines[side].set_visible(False) for side in ['top', 'right']]
            # ax.set_title(mouse)

        s = 1.5 if plot_subgraph else 2
        fig, axs = plt.subplots(1, n_mice, figsize=(s * n_mice, 4.8), sharey=True)
        key = "hub_subgraph" if plot_subgraph else "hub"
        for ax, mouse in zip(axs, valid_mice):
            idxs = {
                "mouse": degree_diffs.mouse == mouse,
                "hub": np.logical_and(
                    degree_diffs.neuron_type == "hub",
                    degree_diffs.trend == "decreasing",
                ),
                "dropped": np.logical_and(
                    degree_diffs.neuron_type == "dropped",
                    degree_diffs.trend == "decreasing",
                ),
                "hub_subgraph": degree_diffs.trend == "subgraph",
            }

            hub = degree_diffs.loc[
                np.logical_and(idxs["mouse"], idxs[key]), "degree_diffs"
            ].values
            dropped = degree_diffs.loc[
                np.logical_and(idxs["mouse"], idxs["dropped"]), "degree_diffs"
            ].values

            data_to_plot = [hub] if plot_subgraph else [hub, dropped]
            xticks = [1] if plot_subgraph else [1, 2]
            xticklabels = (
                ["Hub\nneurons"]
                if plot_subgraph
                else ["Hub\nneurons", "Dropped\nneurons"]
            )
            boxes = ax.boxplot(
                data_to_plot,
                patch_artist=True,
                widths=0.75,
                zorder=0,
                showfliers=False,
            )
            color_boxes(boxes, color)

            x_ = []
            neuron_colors = cycle(["cyan", "orange"])
            colors = ["cyan"] if plot_subgraph else ["cyan", "orange"]
            for i, y in zip(xticks, data_to_plot):
                x = jitter_x(np.ones_like(y), 0.05) * i
                x_.append(x)
                ax.scatter(x, y, color=next(neuron_colors), edgecolors="k")

            if not plot_subgraph:
                ax.plot(x_, data_to_plot, color="k")
            color_boxes(boxes, colors)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, fontsize=14)
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_title(f"Mouse {mouse[0]}")

            if plot_subgraph:
                print(ttest_1samp(hub, 0))
            else:
                print(ttest_rel(hub, dropped))
        axs[0].set_ylabel("Degree difference", fontsize=22)

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.3)

        return Degrees, degree_diffs, fig

    # def R_percentile_comparison(
    #     self,
    #     mouse,
    #     session_type,
    #     n_splits=6,
    #     percentile=33,
    #     plot_all_ensembles=True,
    #     plot_means=True,
    # ):
    #     ensemble_trends = self.find_activity_trends(mouse, session_type)[0]
    #
    #     if not ensemble_trends["decreasing"]:
    #         return None
    #
    #     data = dict()
    #     for ensemble in ensemble_trends["decreasing"]:
    #         data[ensemble] = dict()
    #
    #         xcorr = self.xcorr_ensemble_cells(
    #             mouse, session_type, ensemble, n_splits=n_splits, show_plot=False
    #         )
    #
    #         last_matrix = xcorr["correlations"][-1]
    #         first_matrix = xcorr["correlations"][0]
    #
    #         upper = np.nanpercentile(last_matrix, 100 - percentile)
    #         lower = np.nanpercentile(last_matrix, percentile)
    #
    #         upper_r, upper_c = np.where(last_matrix > upper)
    #         lower_r, lower_c = np.where(last_matrix < lower)
    #
    #         data[ensemble]["upper_first"] = first_matrix[upper_r, upper_c]
    #         data[ensemble]["lower_first"] = first_matrix[lower_r, lower_c]
    #         data[ensemble]["upper_last"] = last_matrix[upper_r, upper_c]
    #         data[ensemble]["lower_last"] = last_matrix[lower_r, lower_c]
    #
    #     if plot_all_ensembles:
    #         ensembles_fig, axs = plt.subplots(
    #             1, len(ensemble_trends["decreasing"]), sharey=True
    #         )
    #         for ax, ensemble in zip(axs, ensemble_trends["decreasing"]):
    #             ax.boxplot(
    #                 [
    #                     data[ensemble][portion]
    #                     for portion in ["upper_first", "lower_first"]
    #                 ],
    #                 widths=0.75,
    #             )
    #
    #         ensembles_fig.tight_layout()
    #
    #     if plot_means:
    #         means_fig, ax = plt.subplots(figsize=(3, 4.8))
    #         self.plot_upper_lower(data, ax, percentile)
    #         ax.set_ylabel("Correlation coefficient")
    #         means_fig.tight_layout()
    #
    #     return data
    #
    # def plot_upper_lower(self, data, ax, percentile=33.33, method="boxplot"):
    #     u = [np.nanmean(xcorrs["upper_first"]) for xcorrs in data.values()]
    #     l = [np.nanmean(xcorrs["lower_first"]) for xcorrs in data.values()]
    #
    #     if method == "boxplot":
    #         boxes = ax.boxplot(
    #             [u, l], widths=0.75, patch_artist=True, zorder=0, showfliers=False
    #         )
    #         color_boxes(boxes, "w")
    #         ax.plot(
    #             [jitter_x(np.ones_like(u), 0.05), jitter_x(np.ones_like(l) * 2, 0.05)],
    #             [u, l],
    #             "ko-",
    #         )
    #         ax.set_xticklabels(
    #             [f"Upper\n{100-percentile}%", f"Lower\n{percentile}%"], rotation=45
    #         )
    #
    #     elif method == "scatter":
    #         ax.scatter(u, l)
    #
    #     else:
    #         raise NotImplementedError
    #     [ax.spines[side].set_visible(False) for side in ["top", "right"]]
    #
    # def plot_delta_by_percentile(self, data, ax, percentile="lower", method="boxplot"):
    #     first = [np.nanmean(xcorrs[percentile + "_first"]) for xcorrs in data.values()]
    #     last = [np.nanmean(xcorrs[percentile + "_last"]) for xcorrs in data.values()]
    #
    #     if method == "boxplot":
    #         boxes = ax.boxplot(
    #             [first, last],
    #             widths=0.75,
    #             patch_artist=True,
    #             zorder=0,
    #             showfliers=False,
    #         )
    #         color_boxes(boxes, "w")
    #
    #         ax.plot(
    #             [
    #                 jitter_x(np.ones_like(first), 0.05),
    #                 jitter_x(np.ones_like(last) * 2, 0.05),
    #             ],
    #             [first, last],
    #             "ko-",
    #         )
    #         ax.set_xticklabels(["Beginning", "End"], rotation=45)
    #
    #     elif method == "scatter":
    #         ax.scatter(first, last)
    #         ax.axis("square")
    #
    #     [ax.spines[side].set_visible(False) for side in ["top", "right"]]
    #
    # def R_percentiles_all_mice(
    #     self,
    #     age="young",
    #     session_type="Reversal",
    #     percentile=33.33,
    #     method="scatter",
    #     **kwargs,
    # ):
    #     """
    #     Sorts cell pair correlation coefficients into the top p percentile and the bottom
    #     1-p percentile based on the correlation matrix computed from the last 5 min of
    #     the session. The top percentile are stable cell pairs and the bottom percentile
    #     are disconnected cell pairs. Then produces two plots.
    #
    #     1) In the FIRST 5 min of the session, the correlation coefficients of the
    #     disconnected cell pairs against the stable cell pairs, one dot per ensemble,
    #     colored by mouse. This shows that within each ensemble, the stable neurons
    #     have higher rhos than the disconnected neurons starting from the beginning of the
    #     session.
    #
    #     2) The correlation coefficients of the disconnected cell pairs in the first
    #     versus the last 5 min of the session. Also the same thing for the stable cell pairs.
    #     This is to show the dynamic range. The disconnected neurons cluster near the bottom
    #     left, meaning they had low rhos to begin with and drop to near 0. In contrast,
    #     the highly correlated cell pairs are stable across time.
    #
    #     :param age:
    #     :param session_type:
    #     :param percentile:
    #     :param method:
    #     :param kwargs:
    #     :return:
    #     """
    #     data = dict()
    #     for mouse in self.meta["grouped_mice"][age]:
    #         data[mouse] = self.R_percentile_comparison(
    #             mouse,
    #             session_type,
    #             percentile=percentile,
    #             plot_all_ensembles=False,
    #             plot_means=False,
    #             **kwargs,
    #         )
    #
    #         if data[mouse] is None:
    #             data.pop(mouse)
    #
    #     if method == "boxplot":
    #         fig, axs = plt.subplots(1, len(data.values()), sharey=True)
    #         for ax, (mouse, values) in zip(axs, data.items()):
    #             self.plot_upper_lower(values, ax, percentile=percentile, method=method)
    #             ax.set_title(mouse)
    #
    #         axs[0].set_ylabel("Correlation coefficient")
    #         fig.tight_layout()
    #         fig.subplots_adjust(wspace=0.3)
    #     elif method == "scatter":
    #         fig, ax = plt.subplots()
    #         for values in data.values():
    #             self.plot_upper_lower(values, ax, percentile=percentile, method=method)
    #         plot_xy_line(ax)
    #         ax.set_xlabel(r"Mean $\rho$ of" "\n" "hub cell pairs")
    #         ax.set_ylabel(r"Mean $\rho$ of" "\n" "dropped cell pairs")
    #         ax.axis("square")
    #         fig.tight_layout()
    #     else:
    #         raise NotImplementedError
    #
    #     if method == "boxplot":
    #         for p in ["lower", "upper"]:
    #             fig, axs = plt.subplots(1, len(data.values()), sharey=True)
    #             for ax, (mouse, values) in zip(axs, data.items()):
    #                 self.plot_delta_by_percentile(
    #                     values, ax, percentile=p, method="boxplot"
    #                 )
    #                 ax.set_title(mouse)
    #
    #             axs[0].set_ylabel("Correlation coefficient")
    #             fig.suptitle(p.capitalize() + " percentiles")
    #             fig.tight_layout()
    #             fig.subplots_adjust(wspace=0.3)
    #
    #     elif method == "scatter":
    #         fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(9, 4.8))
    #         for p, stability, ax in zip(["lower", "upper"], ["dropped", "hub"], axs):
    #             for values in data.values():
    #                 self.plot_delta_by_percentile(
    #                     values, ax, percentile=p, method=method
    #                 )
    #
    #             ax.set_xlabel(
    #                 r"Mean $\rho$ of"
    #                 "\n"
    #                 f"{stability} cell pairs,"
    #                 "\n"
    #                 "beginning of session"
    #             )
    #             ax.set_ylabel(
    #                 r"Mean $\rho$ of"
    #                 "\n"
    #                 f"{stability} cell pairs,"
    #                 "\n"
    #                 "end of session"
    #             )
    #         fig.tight_layout()
    #         fig.subplots_adjust(wspace=0.3)
    #         [plot_xy_line(ax) for ax in axs]
    #
    #     return data

    def connectedness_over_session(
        self,
        mouse,
        session_type,
        n_splits=6,
        method="fdr_bh",
        ax=None,
        color="k",
        metric="degree",
        trend="decreasing",
    ):
        all_ensembles = self.xcorr_fading_ensembles(
            mouse,
            session_type,
            n_splits=n_splits,
            trend=trend,
        )

        if not all_ensembles:
            return None

        G_all_ensembles = [
            self.make_graphs(data, method=method) for data in all_ensembles.values()
        ]

        connectedness = []
        for G in G_all_ensembles:
            if metric == "degree":
                connectedness.append(
                    [np.nanmean([d for n, d in g.degree()]) for g in G]
                )
            elif metric == "clustering_coefficient":
                connectedness.append([average_clustering(g) for g in G])
            else:
                raise NotImplementedError
        connectedness = np.vstack(connectedness)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        # ax.plot(connectedness.T, 'k', alpha=0.1)
        x = np.arange(0, 31, 30 / n_splits).astype(int)
        xtick_labels = [f"{i}-{j} min" for i, j in zip(x[:-1], x[1:])]
        errorfill(
            xtick_labels,
            np.nanmean(connectedness, axis=0),
            yerr=sem(connectedness, axis=0),
            ax=ax,
            color=color,
        )
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        return connectedness

    def connectedness_over_session_all_mice(
        self,
        session_type="Reversal",
        method="fdr_bh",
        n_splits=6,
        age="young",
        metric="clustering_coefficient",
        trend="decreasing",
    ):
        fig, ax = plt.subplots(figsize=(6.4, 6))
        color = age_colors[ages.index(age)]
        connectedness, anova_dfs = dict(), dict()
        ylabel = {
            "clustering_coefficient": "Remodeling ensemble\nclustering coefficient",
            "degree": "Remodeling ensemble\naverage degree",
        }
        for mouse in self.meta["grouped_mice"][age]:
            connectedness[mouse] = self.connectedness_over_session(
                mouse,
                session_type=session_type,
                method=method,
                n_splits=n_splits,
                ax=ax,
                color=color,
                metric=metric,
                trend=trend,
            )
            try:
                anova_dfs[mouse] = pg.rm_anova(
                    pd.DataFrame(connectedness[mouse]), correction=True
                )
            except:
                pass
        ax.set_xlabel("Time (min)", fontsize=22)
        ax.set_ylabel(ylabel[metric], fontsize=22)
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()

        cc = []
        for mouse in self.meta["grouped_mice"][age]:
            try:
                cc.append(np.nanmean(connectedness[mouse], axis=0))
            except:
                pass
        cc = np.vstack(cc)

        anova_dfs["all"] = pg.rm_anova(pd.DataFrame(cc), correction=True)

        return connectedness, anova_dfs, fig

    def intraensemble_activity_over_time(
        self,
        mouse,
        session_type,
        n_splits=6,
        show_plot=True,
        func=np.nanmax,
        data_type="S",
    ):
        """
        Find either the max or mean activity of ensemble members
        across time in a session.

        :param mouse:
        :param session_type:
        :param n_splits:
        :param show_plot:
        :param func:
        :param data_type:
        :return:
        """
        ensemble_trends = self.find_activity_trends(mouse, session_type)[0]
        ylabels = {
            np.nanmax: "Max",
            np.nanmean: "Mean",
        }

        if not ensemble_trends["decreasing"]:
            return None, None

        max_traces = []
        for i, ensemble in enumerate(ensemble_trends["decreasing"]):
            data = self.xcorr_ensemble_cells(
                mouse,
                session_type,
                ensemble,
                n_splits=n_splits,
                show_plot=False,
                data_type=data_type,
            )
            max_traces.append(np.hstack([func(t) for t in data["traces"]]))

        max_traces = pd.DataFrame(np.vstack(max_traces))
        try:
            anova_df = pg.rm_anova(max_traces, correction=True)
        except:
            anova_df = None

        if show_plot:
            fig, ax = plt.subplots()
            ax.plot(max_traces.T, color="k", alpha=0.2)
            errorfill(
                np.arange(max_traces.shape[1]),
                np.nanmean(max_traces, axis=0),
                yerr=sem(max_traces, axis=0),
                ax=ax,
                color="k",
            )

            ax.set_xlabel("Time (normalized)", fontsize=22)
            ax.set_ylabel(f"{ylabels[func]} calcium activity", fontsize=22)
            fig.tight_layout()

        return max_traces, anova_df

    def intraensemble_activity_over_time_all_mice(
        self, session_type, n_splits=6, func=np.nanmax, age="young", data_type="S"
    ):
        """
        Find max or mean of ensemble members across time in a session
        for all mice.

        :param session_type:
        :param n_splits:
        :param func:
        :param age:
        :param data_type:
        :return:
        """
        fig, ax = plt.subplots()
        color = age_colors[ages.index(age)]
        act, anova_dfs = dict(), dict()
        ylabel = {np.nanmax: "Max", np.nanmean: "Mean"}
        for mouse in self.meta["grouped_mice"][age]:
            act[mouse], anova_dfs[mouse] = self.intraensemble_activity_over_time(
                mouse,
                session_type=session_type,
                n_splits=n_splits,
                show_plot=False,
                func=func,
                data_type=data_type,
            )
            try:
                errorfill(
                    np.arange(act[mouse].shape[1]),
                    np.nanmean(act[mouse], axis=0),
                    sem(act[mouse], axis=0),
                    ax=ax,
                    color=color,
                )
            except:
                pass

        ax.set_xlabel("Time (normalized", fontsize=22)
        ax.set_ylabel(f"{ylabel[func]} activity", fontsize=22)
        [ax.spines[side].set_visible(False) for side in ["top", "right"]]
        fig.tight_layout()
        all_act = []
        for mouse in self.meta["grouped_mice"][age]:
            try:
                all_act.append(np.nanmean(act[mouse], axis=0))
            except:
                pass
        all_act = np.vstack(all_act)

        anova_dfs["all"] = pg.rm_anova(pd.DataFrame(all_act), correction=True)

        return act, anova_dfs

    # def find_fading_cluster(self, data, time_bin=-1):
    #     cluster_labels = data["cluster_labels"]
    #     cluster_ids = np.unique(cluster_labels)
    #
    #     corr_mat = data["correlations"][time_bin]
    #     mean_corrs = [np.nanmean(corr_mat[np.where(cluster_labels==cluster)[0]])
    #                   for cluster in cluster_ids]
    #
    #     least_correlated_cluster = np.argmin(mean_corrs)
    #     all_other_clusters = cluster_ids[cluster_ids != least_correlated_cluster]
    #
    #     return least_correlated_cluster, all_other_clusters

    # def compare_connectedness(self, mouse, session_type, n_splits=6):
    #     ensemble_trends = self.find_activity_trends(mouse, session_type)[0]
    #
    #     if not ensemble_trends['decreasing']:
    #         return
    #
    #     data_dict = {
    #         "ensemble": [],
    #         "fading_coeff": [],
    #         "nonfading_coeff": [],
    #     }
    #     for i, ensemble in enumerate(ensemble_trends['decreasing']):
    #         data_dict["ensemble"].append(ensemble)
    #         data = self.xcorr_ensemble_cells(mouse, session_type, ensemble, n_splits=n_splits,
    #                                          show_plot=False)
    #
    #         fading, nonfading = self.find_fading_cluster(data)
    #         corr_mat = data["correlations"][0]
    #         cluster_labels = data["cluster_labels"]
    #
    #         data_dict["fading_coeff"].append(np.nanmean(corr_mat[cluster_labels==fading]))
    #         data_dict["nonfading_coeff"].append(np.nanmean(corr_mat[cluster_labels!=fading]))
    #
    #     fig, ax = plt.subplots()
    #     ax.plot([1,2], [data_dict['fading_coeff'], data_dict['nonfading_coeff']])
    #
    #     print(ttest_rel(data_dict['fading_coeff'], data_dict['nonfading_coeff']))
    #
    #     return data_dict

    def intraensemble_corrs(self, mouse, session_type, n_splits=6):
        ensemble_trends = self.find_activity_trends(mouse, session_type)[0]

        if not ensemble_trends["decreasing"]:
            return

        corr_mats, traces = [], []
        mean_corrs = nan_array((n_splits, len(ensemble_trends["decreasing"])))
        for i, ensemble in enumerate(ensemble_trends["decreasing"]):
            data = self.xcorr_ensemble_cells(
                mouse, session_type, ensemble, n_splits=n_splits, show_plot=True
            )
            corr_mats.append(data["correlations"])
            traces.append(data["traces"])
            mean_corrs[:, i] = [np.nanmean(c) for c in data["correlations"]]

        fig, ax = plt.subplots()
        ax.plot(mean_corrs, "k", alpha=0.2)
        ax.plot(np.nanmean(mean_corrs, axis=1), "k", linewidth=2)

        return mean_corrs

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
            mouse,
            session_types,
            spatial_bin_size_radians=spatial_bin_size_radians,
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
        port_locations_bins = [
            find_reward_spatial_bins(
                lin_position, port_location, spatial_bin_size_radians
            )[0]
            for lin_position, port_location in zip(lin_positions, port_locations)
        ]

        # Sort the fields.
        order = np.argsort(np.argmax(ensemble_fields[sort_by], axis=1))

        # Plot.
        if show_plot:
            session_labels = [
                session.replace("Goals", "Training") for session in session_types
            ]
            if axs is None:
                fig, axs = plt.subplots(1, len(session_types), figsize=(10, 7))

            cmap = matplotlib.cm.get_cmap()
            cmap.set_bad(color='k')
            for ax, fields, ports, session in zip(
                axs, ensemble_fields, port_locations_bins, session_labels
            ):
                ax.imshow(fields[order][subset], cmap=cmap, interpolation='none')

                ax.axis("tight")
                ax.set_title(session)

                ax.set_xticks(ax.get_xlim())
                ax.set_xticklabels([0, 220])

                ax.set_yticks([0, fields.shape[0]-1])
                ax.set_yticklabels([1, fields.shape[0]])
                # yticks = ax.get_yticks().tolist()
                # yticks = [int(ytick + 1) for ytick in yticks]
                # ax.set_yticklabels(yticks)

                for port in ports:
                    ax.axvline(port, c="w")

            axs[0].set_ylabel("Ensemble #")
            fig.suptitle(f"Mouse {mouse[0]}")
            fig.supxlabel("Linearized position (cm)")

        else:
            fig = None

        return ensemble_fields, fig

    def snakeplot_matched_fading_ensembles(self):
        for mouse in self.meta["mice"]:
            ensemble_trends = self.find_activity_trends(mouse, "Reversal")[0]
            self.snakeplot_matched_ensembles(
                mouse,
                ("Goals4", "Reversal"),
                sort_by=1,
                reference_session=1,
                subset=ensemble_trends["decreasing"],
            )

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
            color_boxes(boxes, color)

            if age == "aged":
                ax.set_yticks([])
            else:
                ax.set_ylabel("Assembly spatial information (z)")
            ax.set_xticklabels(mice, rotation=45)

        return SI

    def boxplot_all_assembly_SI(self, ages_to_plot=None, plot_type="line"):
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

        sessions_, ages_, mice_, SIs_ = [], [], [], []
        for session_type in self.meta["session_types"]:
            for age in ages:
                for mouse in self.meta["grouped_mice"][age]:
                    SIs_.append(
                        np.mean(
                            self.data[mouse][session_type]
                            .assemblies["fields"]
                            .data["spatial_info_z"]
                        )
                    )

                    mice_.append(mouse)
                    ages_.append(age)
                    sessions_.append(session_type)

        df = pd.DataFrame(
            {
                "mice": mice_,
                "age": ages_,
                "session": sessions_,
                "SI": SIs_,
            }
        )
        ages_to_plot, plot_colors, n_ages_to_plot = self.ages_to_plot_parser(
            ages_to_plot
        )
        if plot_type == "box":
            fig, axs = plt.subplots(1, len(self.meta["session_types"]), sharey=True)
            fig.subplots_adjust(wspace=0)

            for ax, session_type, title in zip(
                axs, self.meta["session_types"], self.meta["session_labels"]
            ):
                self.scatter_box(SI[session_type], ax=ax, ages_to_plot=ages_to_plot)

                ax.set_xticks([])
                ax.set_title(title, fontsize=14)
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]

            axs[0].set_ylabel("Ensemble spatial info. (z)", fontsize=22)

            if n_ages_to_plot == 2:
                self.set_age_legend(fig)

        else:
            fig, ax = plt.subplots()
            for age, color in zip(ages_to_plot, plot_colors):
                SI_arr = np.vstack([session[age] for session in SI.values()])

                ax.plot(self.meta["session_labels"], SI_arr, color=color, alpha=0.5)
                errorfill(
                    self.meta["session_labels"],
                    np.nanmean(SI_arr, axis=1),
                    sem(SI_arr, axis=1),
                    ax=ax,
                    color=color,
                )
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            ax.set_ylabel("Ensemble spatial info. (z)", fontsize=22)
            fig.tight_layout()

        return SI, df, fig

    def ensemble_SI_anova(self, df):
        SI_anova = pg.rm_anova(df, dv="SI", subject="mice", within="session")
        pairwise_df = df.pairwise_ttests(
            dv="SI", between="session", subject="mice", padjust="sidak"
        )

        return SI_anova, pairwise_df

    def trial_PV_corr(self, mouse, session_type, bin_size=0.05, data_type="ensembles"):
        session = self.data[mouse][session_type]

        if data_type == "ensemble":
            if (
                "rasters" not in session.assemblies["fields"].data
                or session.assemblies["fields"].meta["raster_bin_size"] != bin_size
            ):
                self.make_ensemble_raster(
                    mouse, session_type, bin_size=bin_size, running_only=False
                )

            rasters = session.assemblies["fields"].data["rasters"]
        elif data_type == "cells":
            rasters = session.spatial.data["rasters"]
        else:
            raise NotImplementedError

        n_trials = rasters.shape[1]
        R = nan_array((n_trials, n_trials))
        for combination in product(range(n_trials), repeat=2):
            if combination[0] != combination[1]:
                x = rasters[:, combination[0], :].flatten()
                y = rasters[:, combination[1], :].flatten()

                R[combination[0], combination[1]] = spearmanr(x, y)[0]

        return R

    def plot_all_ensemble_PV_corrs(
        self, age, session_type, midpoint="zero", cmap=matplotlib.cm.bwr
    ):
        fig, axs = plt.subplots(3, 3)
        clims_ = {
            "max": [],
            "min": [],
        }
        R = dict()
        for i, mouse in enumerate(self.meta["grouped_mice"][age]):
            R[mouse] = self.trial_PV_corr(mouse, session_type)

            clims_["max"].append(np.nanmax(R[mouse]))
            clims_["min"].append(np.nanmin(R[mouse]))

            if midpoint == "local":
                axs.flatten()[i].imshow(R[mouse], cmap=cmap)

        if midpoint == "local":
            return R

        clims = {"max": np.nanmax(clims_["max"]), "min": np.nanmin(clims_["min"])}

        if midpoint == "zero":
            midpoint = 1 - clims["max"] / (clims["max"] + abs(clims["min"]))
            cmap = shiftedColorMap(cmap, midpoint=midpoint)

        for i, mouse in enumerate(self.meta["grouped_mice"][age]):
            axs.flatten()[i].imshow(
                R[mouse], vmin=clims["min"], vmax=clims["max"], cmap=cmap
            )

        return R

    def compare_ensemble_PV_corrs(
        self,
        mouse,
        session_types=("Reversal", "Goals4"),
        midpoint="zero",
        cmap=matplotlib.cm.bwr,
    ):
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(mouse)
        clims_ = {
            "max": [],
            "min": [],
        }
        R = dict()
        for ax, session_type in zip(axs, session_types):
            R[session_type] = self.trial_PV_corr(mouse, session_type)

            clims_["max"].append(np.nanmax(R[session_type]))
            clims_["min"].append(np.nanmin(R[session_type]))

            if midpoint == "local":
                ax.imshow(R[session_type], cmap=cmap)

        if midpoint == "local":
            return R

        clims = {"max": np.nanmax(clims_["max"]), "min": np.nanmin(clims_["min"])}
        if midpoint == "zero":
            midpoint = 1 - clims["max"] / (clims["max"] + abs(clims["min"]))
            cmap = shiftedColorMap(cmap, midpoint=midpoint)
        for ax, session_type in zip(axs, session_types):
            ax.imshow(R[session_type], vmin=clims["min"], vmax=clims["max"], cmap=cmap)

        return R

    # def behavior_changepoint(
    #     self,
    #     mouse,
    #     session_type="Reversal",
    #     changepoint_algo=rpt.BottomUp,
    #     behav_trial_threshold=3,
    #     **algo_kwargs,
    # ):
    #     behavior = self.data[mouse][session_type].behavior
    #     (
    #         behavior.data["learning"]["correct_responses"],
    #         behavior.data["learning"]["curve"],
    #         behavior.data["learning"]["start"],
    #         behavior.data["learning"]["inflection"],
    #         behavior.data["learning"]["criterion"],
    #     ) = behavior.get_learning_curve(
    #         trial_threshold=behav_trial_threshold, criterion="individual"
    #     )
    #
    #     changepoint = changepoint_algo(**algo_kwargs).fit_predict(
    #         behavior.data["learning"]["correct_responses"], n_bkps=1
    #     )[0]
    #
    #     return changepoint
    #
    # def ensemble_PV_changepoint(
    #     self,
    #     mouse,
    #     session_type="Reversal",
    #     bin_size=0.05,
    #     changepoint_algo=rpt.BottomUp,
    #     data_type="ensembles",
    #     show_plot=True,
    #     **algo_kwargs,
    # ):
    #     R = self.trial_PV_corr(
    #         mouse, session_type, data_type=data_type, bin_size=bin_size
    #     )
    #
    #     changepoint = changepoint_algo(**algo_kwargs).fit_predict(R, n_bkps=1)[0]
    #
    #     if show_plot:
    #         fig, ax = plt.subplots()
    #
    #         ax.imshow(R, cmap="bwr", aspect="equal")
    #         ax.axvline(x=changepoint)
    #         ax.set_xlabel("Trial")
    #         ax.set_ylabel("Trial")
    #
    #     return changepoint, R
    #
    # def corr_ensemble_PV_and_behavior_changepoints(
    #     self,
    #     age,
    #     session_type="Reversal",
    #     normalize=True,
    #     show_plot=True,
    #     ensemble_function="binned_activations",
    #     **algo_kwargs,
    # ):
    #     f = {
    #         "binned_activations": self.binned_activations_changepoint,
    #         "ensemble_PV": self.ensemble_PV_changepoint,
    #     }
    #     changepoints = pd.DataFrame(columns=["mice", "ensemble", "behavior"])
    #     changepoints_ = dict()
    #     for mouse in self.meta["grouped_mice"][age]:
    #         session = self.data[mouse][session_type]
    #         changepoints_["ensemble"] = f[ensemble_function](
    #             mouse, session_type, show_plot=False, **algo_kwargs
    #         )[0]
    #         changepoints_["behavior"] = self.behavior_changepoint(
    #             mouse, session_type, **algo_kwargs
    #         )
    #
    #         if normalize:
    #             changepoints_ = {
    #                 key: changepoints_[key] / session.behavior.data["ntrials"]
    #                 for key in ["ensemble", "behavior"]
    #             }
    #         changepoints = changepoints.append(
    #             {
    #                 "mice": mouse,
    #                 "ensemble": changepoints_["ensemble"],
    #                 "behavior": changepoints_["behavior"],
    #             },
    #             ignore_index=True,
    #         )
    #
    #     if show_plot:
    #         fig, ax = plt.subplots()
    #         ax.scatter(changepoints["ensemble"], changepoints["behavior"])
    #         ax.set_xlabel("Ensemble changepoint")
    #         ax.set_ylabel("Behavior changepoint")
    #         plot_xy_line(ax)
    #         fig.tight_layout()
    #
    #     return changepoints
    #
    # def binned_activations_changepoint(
    #     self,
    #     mouse,
    #     session_type="Reversal",
    #     changepoint_algo=rpt.BottomUp,
    #     show_plot=True,
    #     **algo_kwargs,
    # ):
    #     trends, binned_activations, slopes, _ = self.find_activity_trends(
    #         mouse,
    #         session_type,
    #         x="trial",
    #         x_bin_size=1,
    #     )
    #     changepoint = changepoint_algo(**algo_kwargs).fit_predict(
    #         binned_activations.T, n_bkps=1
    #     )[0]
    #
    #     if show_plot:
    #         fig, ax = plt.subplots()
    #         ax.imshow(binned_activations)
    #         ax.axvline(x=changepoint)
    #
    #     return changepoint, binned_activations
    #
    # def align_ensemble_peaks_to_behavioral_changepoint(
    #     self,
    #     mouse,
    #     session_type="Reversal",
    #     changepoint_algo=rpt.BottomUp,
    #     xtent=10,
    #     behav_trial_threshold=3,
    #     **algo_kwargs,
    # ):
    #     (
    #         binned_activations_changepoint,
    #         binned_activations,
    #     ) = self.binned_activations_changepoint(
    #         mouse,
    #         session_type,
    #         changepoint_algo=changepoint_algo,
    #         show_plot=False,
    #         **algo_kwargs,
    #     )
    #     changepoints = {
    #         "behavior": self.behavior_changepoint(
    #             mouse,
    #             session_type,
    #             changepoint_algo=changepoint_algo,
    #             behav_trial_threshold=behav_trial_threshold,
    #             **algo_kwargs,
    #         ),
    #         "ensemble": binned_activations_changepoint,
    #     }
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(
    #         np.arange(-xtent, xtent),
    #         np.median(
    #             binned_activations[
    #                 :,
    #                 changepoints["behavior"] - xtent : changepoints["behavior"] + xtent,
    #             ],
    #             axis=0,
    #         ),
    #     )
    #     return changepoints, binned_activations

    def make_fig1(self, panels=None):
        folder = 1
        with open(r"Z:\Will\RemoteReversal\Data\PV_corr_matrices.pkl", "rb") as file:
            corr_matrices = pkl.load(file)

        if panels is None:
            panels = ["A", "C", "D", "E", "F", "G", "H"]

        if "B" in panels:
            age = "young"
            performance_metric = "d_prime"
            df, anova_df, pairwise_df, fig = self.performance_anova(
                age, performance_metric
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"All sessions_{age}_{performance_metric}", folder)

            return anova_df, pairwise_df

        if "C" in panels:
            age = "young"
            performance_metric = "hits"
            df, anova_df, pairwise_df, fig = self.performance_anova(
                age, performance_metric
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"All sessions_{age}_{performance_metric}", folder)

            return anova_df, pairwise_df

        if "D" in panels:
            age = "young"
            performance_metric = "CRs"
            df, anova_df, pairwise_df, fig = self.performance_anova(
                age, performance_metric
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"All sessions_{age}_{performance_metric}", folder)

            return anova_df, pairwise_df

        if "E" in panels:
            metrics = ["hits", "CRs"]
            diffs, fig = self.compare_CR_and_hit_diff(metrics=metrics)

            for metric in metrics:
                print(ttest_1samp(diffs[metric], 0))

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Perf decrease {metric}", folder)

        if "F" in panels:
            fig = self.plot_perseverative_licking(binarize=False)[-1]

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Persev licking", folder)

        if "H" in panels:
            mouse = "Naiad"
            fig = self.plot_max_projs(mouse)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse} max projections", folder)

        if "J" in panels:
            ages_to_plot = "young"
            n_neurons, fig = self.plot_neuron_count(ages_to_plot=ages_to_plot)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Neuron count", folder)

            anova_df = pg.rm_anova(n_neurons.drop(columns="aged"))
            print(anova_df)

            return n_neurons

        if "K" in panels:
            mouse = "Miranda"
            _, fig = self.scrollplot_rasters_by_day(mouse, self.meta["session_types"])

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse}_longitudinal_cell", folder)

        if "L" in panels:
            mouse = "Fornax"
            fig = self.snakeplot_matched_placefields(
                mouse,
                self.meta["session_types"],
                sort_by_session=-2,
                place_cells_only=False,
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse} cells snakeplot", folder)

        if "M" in panels:
            predictors = "cells"
            errors_df, fig = self.plot_spatial_decoder_errors_over_days(
                predictors=predictors
            )[2:]
            anova_df, pairwise_df = self.multisession_spatial_decoder_anova(errors_df)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"Spatial decoder error over sessions_{predictors}", folder
                )

            return anova_df, pairwise_df, errors_df

        if "N" in panels:
            data, df, fig = self.plot_session_PV_corr_comparisons(
                corr_matrices, ages_to_plot="young"
            )
            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Global remapping", folder)

            x = data[("Goals3", "Goals4")]["young"]
            y = data[("Goals4", "Reversal")]["young"]

            print(wilcoxon(x, y))

            return data, df

        if "O" in panels:
            rhos, fig = self.plot_reward_PV_corrs_v2(
                age_to_plot="young", locations_to_plot=["currently_rewarded"]
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Global remapping_newlyrewarded", folder)

            return rhos

        if "P" in panels:
            rhos, fig = self.plot_reward_PV_corrs_v2(
                age_to_plot="young", locations_to_plot=["previously_rewarded"]
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Global remapping_previouslyrewarded", folder)

            return rhos

        if "Q" in panels:
            remap_score_df, fig = self.plot_remap_score_means(
                ages_to_plot="young", place_cells_only=False
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Rate remapping", folder)

            return remap_score_df

        if "R" in panels:
            remap_score_df, fig = self.plot_remap_score_means(
                ages_to_plot="young",
                place_cells_only=False,
                ports=["original rewards", "newly rewarded"],
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Rate remapping_newlyrewarded", folder)

            return remap_score_df

        if "S" in panels:
            remap_score_df, fig = self.plot_remap_score_means(
                ages_to_plot="young",
                place_cells_only=False,
                ports=["original rewards", "previously rewarded"],
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Rate remapping_previouslyrewarded", folder)

            return remap_score_df

    def make_fig2(self, panels=None):
        folder = 2
        if panels is None:
            panels = ["A", "B", "C", "D", "E"]

        if "A" in panels:
            mouse = "Fornax"
            session_type = "Goals4"
            ensembles = [0, 1, 2, 3]

            fig = self.plot_ex_patterns(mouse, session_type, ensembles)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse} ensemble patterns {session_type}", folder)

        if "B" in panels:
            mouse = "Lyra"
            session_type = "Goals4"
            ensemble = 38
            fig = self.plot_ensemble(mouse, session_type, ensemble)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"{mouse} ensemble {ensemble} from {session_type}", folder
                )

        if "C" in panels:
            mouse = "Lyra"
            session_type = "Goals4"
            ensemble = 38
            fig = self.plot_ensemble_raster(mouse, session_type, ensemble, bin_size=0.2)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"{mouse}_{session_type}_ensemble{ensemble}_raster", folder
                )

        if "D" in panels:
            n_ensembles, df, fig = self.count_ensembles(ages_to_plot="young")

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "# ensembles", folder)

            anova_df = pg.rm_anova(df.drop(columns="aged"))
            print(anova_df)

            return df

        if "E" in panels:
            errors_df, fig = self.plot_spatial_decoder_errors_over_days()[2:]
            anova_df, pairwise_df = self.multisession_spatial_decoder_anova(errors_df)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Spatial decoder error over sessions", folder)

            return anova_df, pairwise_df, errors_df

        if "F" in panels:
            mouse = "Miranda"
            ensemble_fields, fig = self.snakeplot_matched_ensembles(
                mouse, ("Goals3", "Goals4")
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse}_ensemble_snakeplot", folder)

        if "G" in panels:
            ages_to_plot = "young"
            lag = 0
            df, fig = self.plot_lick_decoder(
                licks_to_include="first",
                lag=lag,
                ages_to_plot=ages_to_plot,
                class_weight="balanced",
                random_state=7,
                n_jobs=6,
                shuffle=True,
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"EnsembleLickDecoding_{ages_to_plot}_lag{lag}", folder
                )

            return df

    def make_fig3(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        folder = 3

        if "A" in panels:
            mouse = "Lyra"
            session_type = "Reversal"
            ensemble = 56
            fig = self.plot_ensemble(mouse, session_type, ensemble)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse}_{session_type}_ensemble{ensemble}", folder)

            fig = self.plot_ensemble_raster(mouse, session_type, ensemble, bin_size=0.2)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"{mouse}_{session_type}_ensemble{ensemble}_raster", folder
                )

            fig = self.plot_activation_trend(mouse, session_type, ensemble)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"{mouse}_max_activity_per_trial_ensemble_{ensemble}", folder
                )

        if "B" in panels:
            mouse = "Lyra"
            session_type = "Reversal"
            ensemble = 58
            fig = self.plot_ensemble(mouse, session_type, ensemble)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse}_{session_type}_ensemble{ensemble}", folder)

            fig = self.plot_ensemble_raster(mouse, session_type, ensemble, bin_size=0.2)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"{mouse}_{session_type}_ensemble{ensemble}_raster", folder
                )

            fig = self.plot_activation_trend(mouse, session_type, ensemble)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"{mouse}_max_activity_per_trial_ensemble_{ensemble}", folder
                )

        if "C" in panels:
            ages_to_plot = "young"
            z_threshold = None
            p_changing_split_by_age, fig = self.plot_proportion_changing_units(
                ages_to_plot=ages_to_plot, z_threshold=z_threshold
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Percent_fading_{ages_to_plot}", folder)

            stats = wilcoxon(
                p_changing_split_by_age["young"][1],
                p_changing_split_by_age["young"][0],
                zero_method="zsplit",
            )
            W = np.round(stats.statistic, 3)
            p = np.round(stats.pvalue, 3)

            msg = f"Young: W: {W}, p = {p}"
            if p < 0.05:
                msg += "*"
            print(msg)

        if "D" in panels:
            z_threshold = None
            performance_metric = "CRs"
            fig = self.correlate_prop_changing_ensembles_to_behavior(
                performance_metric=performance_metric,
                ages_to_plot="young",
                z_threshold=z_threshold,
            )[-1]

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"FadingEnsembleCorr_{performance_metric}_young", folder
                )

        if "E" in panels:
            fig = self.correlate_fade_slope_to_learning_slope(
                age_to_plot="young", dv="reg_slope", performance_metric="CRs"
            )[1]

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig,
                    "Correlation of fading ensemble slope to learning slope",
                    folder,
                )

        if "F" in panels:
            mouse = "Fornax"
            ensemble_number = 54
            fig, cbar_fig = self.plot_graph_evolution(
                mouse, "Reversal", ensemble_number, plot_cbar=True
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig,
                    f"Cell correlations for {mouse}, ensemble #{ensemble_number}",
                    folder,
                )
                self.save_fig(cbar_fig, f"cbar", folder)

        if "G" in panels:
            age = "young"
            metric = "clustering_coefficient"
            cluster_coeffs, anova_dfs, fig = self.connectedness_over_session_all_mice(
                age=age, metric=metric
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{metric}  of ensemble members_{age}", folder)

            return anova_dfs

    def make_fig4(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C", "D"]

        folder = 4

        if "A" in panels:
            age = "young"
            metric = "degree"
            cluster_coeffs, anova_dfs, fig = self.connectedness_over_session_all_mice(
                age=age, metric=metric
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{metric} of ensemble members_{age}", folder)

            return anova_dfs

        if "B" in panels:
            mouse = "Fornax"
            n_splits = 6
            fig = self.plot_degree_distribution(mouse, "Reversal", 8, n_splits=n_splits)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Degree distribution {mouse}", folder)

        if "C" in panels:
            mouse = "Fornax"
            ensembles = [8, 35, 52, 54]
            n_splits = 6
            fig, axs = plt.subplots(len(ensembles), n_splits, figsize=(9, 4.8))

            for ensemble, ax in zip(ensembles, axs):
                self.plot_graphs_deg_labels(mouse, "Reversal", ensemble, axs=ax)

            x = np.arange(0, 31, 30 / n_splits).astype(int)
            titles = [f"{i}-{j} min" for i, j in zip(x[:-1], x[1:])]
            for ax, title in zip(axs[0], titles):
                ax.set_title(title, fontsize=14)
            fig.supylabel(f"Remodeling ensembles\nfrom {mouse}", fontsize=22)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Spring graph labeled {mouse} {ensembles}", folder)

        if "D" in panels:
            age = "young"
            Degrees, fig, t_tests = self.degree_comparison_all_mice(age=age)

            for mouse, result in t_tests.items():
                print(f"{mouse}: {result}")

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Hub vs disconnected degrees", folder)

            return Degrees

        # if "C" in panels:
        #     age = "young"
        #
        #     for mouse in self.meta['grouped_mice'][age]:
        #         fig = self.degree_start_end_session(mouse, "Reversal", quartiles_to_plot=[1, 4])[-1]
        #
        #         if fig is not None:
        #             fig.suptitle(mouse)
        #
        #             if self.save_configs["save_figs"]:
        #                 self.save_fig(fig, f"{mouse} degrees over time", folder)

        if "E" in panels:
            act_rate_df = pd.read_csv(
                os.path.join(
                    self.save_configs["path"], "S7", "all_sessions_activity_rate.csv"
                )
            )
            session_ids = act_rate_df["session_id"].unique()

            fig, axs = plt.subplots(
                1,
                len(session_ids),
                figsize=(2 * len(session_ids), 4.8),
                sharey=True,
                sharex=True,
            )
            for session_id, ax in zip(session_ids, axs):
                # boxes = ax.boxplot(
                #     [
                #         act_rate_df.loc[np.logical_and(
                #             act_rate_df['session_id']==session_id,
                #             act_rate_df['category']==category), 'event_rate']
                #         for category in categories
                #     ], widths=0.75, patch_artist=True,
                # )
                # color_boxes(boxes, age_colors[0])
                # ax.set_xticklabels(['Hub\nneurons', 'Dropped\nneurons'], rotation=45)

                data = act_rate_df.loc[act_rate_df["session_id"] == session_id]
                data.drop_duplicates(["neuron_id", "mouse"], inplace=True)
                sns.kdeplot(
                    data=data,
                    x="event_rate",
                    hue="category",
                    fill=True,
                    palette={"dropped": "orange", "hub": "limegreen"},
                    ax=ax,
                    legend=False,
                )
                ax.set_title(session_id.replace("Goals", "Training"))
                ax.set_xlabel("")
                [ax.spines[side].set_visible(False) for side in ["top", "right"]]
            axs[0].set_ylabel("Density", fontsize=22)
            axs[0].set_yticks(axs[0].get_ylim())
            axs[0].set_yticklabels([0, 1])
            fig.supxlabel("Ca2+ transient rate")
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.1)

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig,
                    "Activity rate of hub and dropped neurons over sessions",
                    folder,
                )

            return act_rate_df

    def format_Degrees_for_Mark(self, Degrees, fname):
        """
        Format the Degrees DataFrame from self.degree_comparison_all_mice() to a clean version,
        and save as a csv.

        """
        df = Degrees.copy()
        df = df.loc[df["time_bin"] == 0]
        df = df.drop(columns="time_bin")
        df = df.loc[df["quartile"].isin([1, 4])]
        df.loc[df["quartile"] == 1, "quartile"] = "dropped"
        df.loc[df["quartile"] == 4, "quartile"] = "hub"
        df = df.rename(columns={"quartile": "category"})
        df = df[["mouse", "ensemble_id", "neuron_id", "category", "degree"]]
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.save_configs["path"], "4", fname))

        return df

    def make_fig5(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C", "D", "E", "F", "G"]

        folder = 5

        if "A" in panels:
            anova_df = self.aged_performance_anova(show_plot=False)
            print(anova_df)
            self.plot_peak_performance_all_sessions(performance_metric="d_prime", plot_line=True,
                                                  sessions=["Goals"+str(i) for i in np.arange(1,5)])

        if "B" in panels:
            performance_metric = "CRs"
            dv, anova_dfs, fig, sessions_df = self.plot_trial_behavior(
                session_types=["Goals4", "Reversal"],
                performance_metric=performance_metric,
            )
            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"Aged vs young Reversal_{performance_metric}", folder
                )

            sessions_df.to_csv(
                os.path.join(
                    self.save_configs["path"],
                    f"{folder}",
                    f"age_{performance_metric}.csv",
                )
            )

            for df in anova_dfs.values():
                print(df)

            performance, fig = self.plot_performance_session_type(
                "Reversal",
                window=None,
                strides=None,
                performance_metric=performance_metric,
            )
            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"Reversal_aged_vs_young_{performance_metric}", folder
                )

            stats = ttest_ind(performance["young"], performance["aged"])
            p = np.round(stats.pvalue, 3)
            t = np.round(stats.statistic, 3)
            dof = len(performance["young"]) + len(performance["aged"]) - 2
            msg = f"Young vs Aged on Reversal: t({dof})={t}, p={p}"
            if p < 0.05:
                msg += "*"
            print(msg)

        if "C" in panels:
            ages_to_plot = None
            z_threshold = None
            p_changing_split_by_age, fig = self.plot_proportion_changing_units(
                ages_to_plot=ages_to_plot, z_threshold=z_threshold
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Percent_fading_{ages_to_plot}", folder)

            stats = wilcoxon(
                p_changing_split_by_age["aged"][1],
                p_changing_split_by_age["aged"][0],
                zero_method="zsplit",
            )
            W = np.round(stats.statistic, 3)
            p = np.round(stats.pvalue, 3)

            msg = f"Aged: W: {W}, p = {p}"
            if p < 0.05:
                msg += "*"
            print(msg)

        if "D" in panels:
            z_threshold = None
            performance_metric = "CRs"
            fig = self.correlate_prop_changing_ensembles_to_behavior(
                performance_metric=performance_metric,
                ages_to_plot=None,
                z_threshold=z_threshold,
            )[-1]

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"FadingEnsembleCorr_{performance_metric}_aged", folder
                )

    def make_sfig1(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C", "D", "E"]

        folder = "S2"
        if "A" in panels:
            performance_metric = "d_prime"
            fig = self.plot_reversal_vs_training4_trial_behavior(
                "young", performance_metric
            )[1]

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig,
                    f"Training vs Reversal performance_{performance_metric}",
                    folder,
                )

        if "B" in panels:
            performance_metric = "hits"
            fig = self.plot_reversal_vs_training4_trial_behavior(
                "young", performance_metric
            )[1]

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig,
                    f"Training vs Reversal performance_{performance_metric}",
                    folder,
                )

        if "C" in panels:
            performance_metric = "CRs"
            fig = self.plot_reversal_vs_training4_trial_behavior(
                "young", performance_metric
            )[1]

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig,
                    f"Training vs Reversal performance_{performance_metric}",
                    folder,
                )

        if "D" in panels:
            pers, reg, fig = self.plot_perseverative_licking_over_session(
                binarize_licks=False
            )

            for df, name in zip([pers, reg], ["persev", "reg"]):
                df = pd.DataFrame(df["young"])
                df["mice"] = self.meta["grouped_mice"]["young"]
                df = df.set_index("mice")
                df = df.stack().reset_index()
                df = df.rename(columns={"level_1": "trial_bin", 0: "licks"})

                df.to_csv(
                    os.path.join(
                        self.save_configs["path"], folder, f"{name}_licking.csv"
                    )
                )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Perseverative licking", folder)

        if "E" in panels:
            fig = self.plot_all_reward_field_densities(
                ["Goals3", "Goals4", "Reversal"], ages_to_plot="young"
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Reward field density", folder)

    def make_sfig3(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C", "D", "E", "F", "G"]
        folder = "S3"
        mouse_panels = {
            panel: mouse
            for panel, mouse in zip(panels, self.meta["grouped_mice"]["young"])
        }

        for panel in panels:
            mouse = mouse_panels[panel]

            fig = self.snakeplot_matched_placefields(
                mouse,
                self.meta["session_types"],
                sort_by_session=-2,
                place_cells_only=False,
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse}_snakeplots", folder)

    def make_sfig4(self, panels=None):
        if panels is None:
            panels = ["A", "B"]

        folder = "S4"
        if "A" in panels:
            ages_to_plot = "young"
            df, fig = self.boxplot_all_assembly_SI(ages_to_plot, plot_type="line")[1:]
            SI_anova, pairwise_df = self.ensemble_SI_anova(df)

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Ensemble_spatial_info_{ages_to_plot}", folder)

            return SI_anova, pairwise_df

        if "B" in panels:
            data, ensemble_size_df, fig = self.plot_ensemble_sizes(
                data_type="ensemble_size", ages_to_plot="young"
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Ensemble sizes", folder)

            # anova_df = pg.rm_anova(ensemble_size_df)
            # print(anova_df)
            pt = pd.pivot_table(
                ensemble_size_df.loc[ensemble_size_df["age"] == "young"],
                values="ensemble_size",
                index="mice",
                columns="session",
            )
            anova_df = pg.rm_anova(pt)
            print(anova_df)

            return ensemble_size_df

    def make_sfig5(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C", "D"]
        folder = "S5"
        if "A" in panels:
            self.plot_ensemble_registration_ex()

        if "B" in panels:
            mouse = "Miranda"
            subset = [11]
            fig = self.plot_matched_ensembles(mouse, ("Goals3", "Goals4"), subset=[11])

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{mouse}_Ensemble{subset}_matched", folder)

        if "C" in panels:
            mouse = "Miranda"
            session_types = ["Goals3", "Goals4"]
            ensembles = [11, 10]

            for session_type, ensemble in zip(session_types, ensembles):
                fig = self.plot_ensemble_raster(
                    mouse, session_type, ensemble, bin_size=0.2
                )

                if self.save_configs["save_figs"]:
                    self.save_fig(
                        fig, f"{mouse}_{session_type}_ensemble_raster", folder
                    )

        if "D" in panels:
            ages_to_plot = "young"
            lag = -1
            df, fig = self.plot_lick_decoder(
                licks_to_include="first",
                lag=lag,
                ages_to_plot=ages_to_plot,
                class_weight="balanced",
                random_state=7,
                n_jobs=6,
                shuffle=True,
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"EnsembleLickDecoding_{ages_to_plot}_lag{lag}", folder
                )

            return df

    def make_sfig6(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C"]

        folder = "S6"

        if "A" in panels:
            ages_to_plot = None
            p_changing_split_by_age, fig = self.plot_proportion_changing_units(
                data_type="S", ages_to_plot=ages_to_plot
            )
            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Percent_fading_{ages_to_plot}", folder)

            for age in ages:
                stats = wilcoxon(
                    p_changing_split_by_age[age][1],
                    p_changing_split_by_age[age][0],
                    zero_method="zsplit",
                )
                W = np.round(stats.statistic, 3)
                p = np.round(stats.pvalue, 3)

                msg = f"{age}: W: {W}, p = {p}"
                if p < 0.05:
                    msg += "*"
                print(msg)

        if "B" in panels:
            ages_to_plot = "young"
            _, _, chi, fig = self.plot_proportion_fading_ensembles_near_important_ports(
                ages_to_plot=ages_to_plot
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"Fading ensemble field location_{ages_to_plot}", folder
                )

            return chi

        if "C" in panels:
            cluster_coeffs, anova_dfs, fig = self.connectedness_over_session_all_mice(
                age="young", metric="clustering_coefficient", trend="no trend"
            )

            print(anova_dfs)
            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"Non-remodeling ensemble CC", folder)

    def make_sfig7(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C"]

        folder = "S7"
        if "A" in panels:
            Degrees, fig = self.contour_degree_beginning_and_end("young")
            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Hub vs disconnected degree diff contour", folder)

        if "B" in panels:
            fig = self.delta_degree_comparison("young")[-1]

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Hub vs disconnected degree diff", folder)

        if "C" in panels:
            Degrees, degree_diffs, fig = self.delta_degree_comparison(
                age="young", plot_subgraph=True
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, "Hub vs dropped neuron delta deg_no dropped neurons", folder
                )

    def make_sfig8(self, panels=None):
        if panels is None:
            panels = ["A", "B"]

        folder = "S8"

        if "A" in panels:
            master_df = pd.DataFrame()
            for session_type in self.meta["session_types"]:
                print(f"{session_type}: ")
                df, fig = self.hub_dropped_cells_metrics_all_mice(
                    self.meta["grouped_mice"]["young"][:-2],
                    "Reversal",
                    activity_rate_session=session_type,
                    graph_session_neuron_id=True,
                    half=None,
                    metric="event_rate",
                )

                # Save fig and csv.
                if self.save_configs["save_figs"]:
                    self.save_fig(
                        fig,
                        f"Hub vs dropped neurons transient rate {session_type}",
                        folder,
                    )

                df.to_csv(
                    os.path.join(
                        self.save_configs["path"],
                        folder,
                        f"{session_type}_act_rate.csv",
                    )
                )

                master_df = pd.concat((master_df, df))
                print("")

            master_df.to_csv(
                os.path.join(
                    self.save_configs["path"], folder, "all_sessions_activity_rate.csv"
                )
            )

            for half, half_i in zip(["first half", "second half"], [0, 1]):
                print(f"{half}:")
                df, fig = self.hub_dropped_cells_metrics_all_mice(
                    self.meta["grouped_mice"]["young"][:-2],
                    "Reversal",
                    graph_session_neuron_id=True,
                    half=half_i,
                    metric="event_rate",
                )
                print("")

                if self.save_configs["save_figs"]:
                    self.save_fig(
                        fig,
                        f"Hub vs dropped neurons transient rate reversal {half}",
                        folder,
                    )

                df.to_csv(
                    os.path.join(
                        self.save_configs["path"],
                        folder,
                        f"reversal_{half}_act_rate.csv",
                    )
                )
        #
        # if "B" in panels:
        #     master_df = pd.DataFrame()
        #     for session_type in self.meta["session_types"]:
        #         print(f"{session_type}: ")
        #         df, fig = self.hub_dropped_cells_metrics_all_mice(
        #             self.meta["grouped_mice"]["young"][:-2],
        #             "Reversal",
        #             activity_rate_session=session_type,
        #             graph_session_neuron_id=True,
        #             half=None,
        #             metric="spatial_info",
        #         )
        #
        #         # Save fig and csv.
        #         if self.save_configs["save_figs"]:
        #             self.save_fig(
        #                 fig,
        #                 f"Hub vs dropped neurons spatial info {session_type}",
        #                 folder,
        #             )
        #
        #         df.to_csv(
        #             os.path.join(
        #                 self.save_configs["path"],
        #                 folder,
        #                 f"{session_type}_act_rate.csv",
        #             )
        #         )
        #
        #         master_df = pd.concat((master_df, df))
        #         print("")
        #
        #     master_df.to_csv(
        #         os.path.join(
        #             self.save_configs["path"], folder, "all_sessions_spatial_info.csv"
        #         )
        #     )

    def make_sfig9(self, panels=None):
        if panels is None:
            panels = ["A", "B", "C", "D", "E", "F", "G", "H"]

        folder = "S9"

        if "A" in panels:
            trials, fig = self.compare_trial_count("Reversal")

            print(ttest_ind(trials["young"], trials["aged"]))

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Trial counts", folder)

        if "B" in panels:
            licks, fig = self.compare_licks("Reversal", exclude_rewarded=False)

            print(ttest_ind(licks["young"], licks["aged"]))

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Lick counts", folder)

        if "C" in panels:
            licks, fig = self.compare_licks("Reversal", exclude_rewarded=True)

            print(ttest_ind(licks["young"], licks["aged"]))

            if self.save_configs["save_figs"]:
                self.save_fig(fig, "Lick counts_exclude_rewarded", folder)

        if "D" in panels:
            performance_metric = "d_prime"
            dv, anova_dfs, fig, sessions_df = self.plot_trial_behavior(
                session_types=["Goals4", "Reversal"],
                performance_metric=performance_metric,
                window=6,
                strides=2,
            )

            sessions_df.to_csv(
                os.path.join(
                    self.save_configs["path"],
                    f"{folder}",
                    f"age_{performance_metric}.csv",
                )
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"Young vs aged Reversal_{performance_metric}", folder
                )

            for df in anova_dfs.values():
                print(df)

        if "E" in panels:
            performance_metric = "hits"
            dv, anova_dfs, fig, sessions_df = self.plot_trial_behavior(
                session_types=["Goals4", "Reversal"],
                performance_metric=performance_metric,
                window=6,
                strides=2,
            )

            sessions_df.to_csv(
                os.path.join(
                    self.save_configs["path"],
                    f"{folder}",
                    f"age_{performance_metric}.csv",
                )
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig, f"Young vs aged Reversal_{performance_metric}", folder
                )

            for df in anova_dfs.values():
                print(df)

        if "F" in panels:
            performance_metric = "d_prime"
            fig, axs = plt.subplots(1, 2, figsize=(6, 4.75), sharey=True)
            for ax, downsample, title in zip(
                axs, [False, True], ["All trials", "Downsampled"]
            ):
                bp, fig = self.plot_performance_session_type(
                    "Reversal",
                    performance_metric=performance_metric,
                    downsample_trials=downsample,
                    ax=ax,
                )
                ax.set_title(title)
                print(title)
                print(ttest_ind(*[value for value in bp.values()]))
            axs[-1].set_ylabel("")
            print("")

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{performance_metric} downsamp", folder)

        if "G" in panels:
            performance_metric = "hits"
            fig, axs = plt.subplots(1, 2, figsize=(6, 4.75), sharey=True)
            for ax, downsample, title in zip(
                axs, [False, True], ["All trials", "Downsampled"]
            ):
                bp, fig = self.plot_performance_session_type(
                    "Reversal",
                    performance_metric=performance_metric,
                    downsample_trials=downsample,
                    ax=ax,
                )
                ax.set_title(title)
                print(title)
                print(ttest_ind(*[value for value in bp.values()]))
            axs[-1].set_ylabel("")
            print("")

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{performance_metric} downsamp", folder)

        if "H" in panels:
            performance_metric = "CRs"
            fig, axs = plt.subplots(1, 2, figsize=(6, 4.75), sharey=True)
            for ax, downsample, title in zip(
                axs, [False, True], ["All trials", "Downsampled"]
            ):
                bp, fig = self.plot_performance_session_type(
                    "Reversal",
                    performance_metric=performance_metric,
                    downsample_trials=downsample,
                    ax=ax,
                )
                ax.set_title(title)
                print(title)
                print(ttest_ind(*[value for value in bp.values()]))
            axs[-1].set_ylabel("")
            print("")

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{performance_metric} downsamp", folder)

    def make_sfig10(self, panels=None):
        if panels is None:
            panels = ["A", "B"]
        folder = "S8"

        if "A" in panels:
            mouse = "Umbriel"
            ensemble_number = 1
            fig, cbar_fig = self.plot_graph_evolution(
                mouse, "Reversal", ensemble_number, plot_cbar=True
            )

            if self.save_configs["save_figs"]:
                self.save_fig(
                    fig,
                    f"Cell correlations for {mouse}, ensemble #{ensemble_number}",
                    folder,
                )
                self.save_fig(cbar_fig, f"cbar", folder)

        if "B" in panels:
            age = "aged"
            metric = "clustering_coefficient"
            cluster_coeffs, anova_dfs, fig = self.connectedness_over_session_all_mice(
                age=age, metric=metric
            )

            if self.save_configs["save_figs"]:
                self.save_fig(fig, f"{metric} of ensemble members_{age}", folder)

            return anova_dfs

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
            "Janus",
            "Lyra",
            "Miranda",
            "Naiad",
            "Oberon",
            "Puck",
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
