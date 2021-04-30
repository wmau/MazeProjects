import matplotlib.pyplot as plt
from CaImaging.util import (
    sem,
    nan_array,
    bin_transients,
    make_bins,
    ScrollPlot,
    contiguous_regions,
)
from CaImaging.plotting import errorfill, beautify_ax
from scipy.stats import spearmanr, zscore, circmean
from CircleTrack.SessionCollation import MultiAnimal
from CaImaging.CellReg import rearrange_neurons, trim_map, scrollplot_footprints
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import os
from CircleTrack.plotting import plot_daily_rasters, spiral_plot, highlight_column
from CaImaging.Assemblies import preprocess_multiple_sessions, lapsed_activation
from CircleTrack.Assemblies import plot_assembly, \
    spatial_bin_ensemble_activations
import xarray as xr
import pymannkendall as mk
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from CaImaging.PlaceFields import spatial_bin
from tqdm import tqdm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from CircleTrack.utils import get_circular_error, format_spatial_location_for_decoder

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
}

aged_mice = [
    "Gemini",
    "Oberon",
    "Puck",
]


class ProjectAnalyses:
    def __init__(self, mice, project_name="RemoteReversal", behavior_only=False):
        # Collect data from all mice and sessions.
        self.data = MultiAnimal(
            mice, project_name=project_name, behavior_only=behavior_only
        )

        # Define session types here. Watch out for typos.
        # Order matters. Plots will be in the order presented here.
        self.meta = {
            "session_types": session_types[project_name],
            "mice": mice,
        }

        self.meta["session_labels"] = [
            session_type.replace("CircleTrack", "")
            for session_type in self.meta["session_types"]
        ]

        self.meta["grouped_mice"] = {
            "aged": [mouse for mouse in self.meta["mice"] if mouse in aged_mice],
            "young": [mouse for mouse in self.meta["mice"] if mouse not in aged_mice],
        }

        self.meta["aged"] = {
            mouse: True if mouse in aged_mice else False for mouse in self.meta["mice"]
        }

        # for mouse in mice:
        #     S_list = [self.data[mouse][session].data['imaging']['S']
        #               for session in self.meta['session_types]]
        #
        #     cell_map = self.data[mouse]['CellReg'].map
        #
        #     rearranged = rearrange_neurons(cell_map, S_list)

    def count_ensembles(self):
        n_ensembles = nan_array(
            (len(self.meta["mice"]), len(self.meta["session_types"]))
        )
        for i, mouse in enumerate(self.meta["mice"]):
            for j, session_type in enumerate(self.meta["session_types"]):
                session = self.data[mouse][session_type]
                n_ensembles[i, j] = session.assemblies["significance"].nassemblies

        fig, ax = plt.subplots()
        for n, mouse in zip(n_ensembles, self.meta["mice"]):
            color = "k" if mouse in self.meta["grouped_mice"]["young"] else "r"
            ax.plot(self.meta["session_labels"], n, color=color)
            ax.annotate(mouse, (0.1, n[0] + 1))

        ax.set_ylabel("# of ensembles")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        fig.subplots_adjust(bottom=0.2)

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
            'S' or 'C').
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

    def find_all_overlaps(self, show_plot=True):
        """
        Wrapper function for find_percent_overlap, run this per mouse.

        """
        n_sessions = len(self.meta["session_types"])
        overlaps = nan_array((len(self.meta["mice"]), n_sessions, n_sessions))

        for i, mouse in enumerate(self.meta["mice"]):
            overlaps[i] = self.find_percent_overlap(mouse, show_plot=False)

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

    def find_percent_overlap(self, mouse, show_plot=True):
        n_sessions = len(self.meta["session_types"])
        r, c = np.indices((n_sessions, n_sessions))
        overlaps = np.ones((n_sessions, n_sessions))
        for session_pair, i, j in zip(
            product(self.meta["session_types"], self.meta["session_types"]),
            r.flatten(),
            c.flatten(),
        ):
            if i != j:
                overlap_map = self.get_cellreg_mappings(
                    mouse, session_pair, detected="first_day"
                )[0]

                n_neurons_reactivated = np.sum(overlap_map.iloc[:, 1] != -9999)
                n_neurons_first_day = len(overlap_map)
                overlaps[i, j] = n_neurons_reactivated / n_neurons_first_day

        if show_plot:
            fig, ax = plt.subplots()
            ax.imshow(overlaps)

        return overlaps

    def plot_registered_cells(self, mouse, session_types, neurons_from_session1=None):
        session_list = self.get_cellreg_mappings(
            mouse, session_types, neurons_from_session1=neurons_from_session1
        )[-1]

        scrollplot_footprints(
            self.data[mouse]["CellReg"].path, session_list, neurons_from_session1
        )

    def map_placefields(self, mouse, session_types, neurons_from_session1=None):
        # Get neurons and cell registration mappings.
        trimmed_map, global_idx = self.get_cellreg_mappings(
            mouse, session_types, neurons_from_session1=neurons_from_session1
        )[:-1]

        # Get fields.
        fields = {
            session_type: self.data[mouse][session_type].spatial.data["placefields"]
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

    def plot_goals_vs_reversal_stability(self):
        goals = ("Goals3", "Goals4")
        reversal = ("Goals4", "Reversal")

        median_r = np.zeros((len(self.meta["mice"]), 2))
        fig, ax = plt.subplots()
        for i, mouse in enumerate(self.meta["mice"]):
            median_r[i, 0] = np.nanmedian(
                self.correlate_fields(mouse, goals, show_histogram=False)["r"]
            )
            median_r[i, 1] = np.nanmedian(
                self.correlate_fields(mouse, reversal, show_histogram=False)["r"]
            )

            ax.plot(median_r[i], "o-")

    def plot_aged_pf_correlation(self, session_types=("Goals3", "Goals4")):

        r_values = dict()
        for age in ["young", "aged"]:
            r_values[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                rs_this_mouse = np.asarray(
                    self.correlate_fields(mouse, session_types, show_histogram=False)[
                        "r"
                    ]
                )
                rs = rs_this_mouse[~np.isnan(rs_this_mouse)]

                r_values[age].append(rs)

        fig, axs = plt.subplots(1, 2, sharey=True)
        fig.subplots_adjust(wspace=0)

        for ax, age in zip(axs, ["young", "aged"]):
            ax.violinplot(r_values[age], showmedians=True, showextrema=False)
            ax.set_title(age)
            ax.set_xticks(np.arange(1, len(self.meta["grouped_mice"][age]) + 1))
            ax.set_xticklabels(self.meta["grouped_mice"][age], rotation=45)

            if age == "young":
                ax.set_ylabel("Place field correlations [r]")

        return r_values

    def get_placefield_distribution_comparisons(
        self,
        session_pairs=(
            ("Goals3", "Goals4"),
            ("Goals4", "Reversal"),
        ),
    ):
        r = {key: [] for key in session_pairs}
        for mouse in self.meta["mice"]:
            for key, pair in zip(session_pairs, session_pairs):
                r[key].append(
                    self.correlate_fields(mouse, pair, show_histogram=False)["r"]
                )

        fig, ax = plt.subplots()
        for i, data in enumerate(r.values()):
            x = np.linspace(2 * i, 2 * i + 1, len(self.meta["mice"]))
            plot_me = [
                np.asarray(mouse_data)[~np.isnan(mouse_data)] for mouse_data in data
            ]
            ax.boxplot(plot_me, positions=x)

        ax.set_ylabel("Firing field correlations [r]")

        return r

    def plot_rasters_by_day(
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
                    sessions[session_type].spatial.data["placefields"][
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
            )

        # List of dictionaries. Do HoloMap(daily_rasters) in a
        # jupyter notebook.
        return daily_rasters

    def decode_place(
        self,
        mouse,
        training_and_test_sessions,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        show_plot=True,
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
        S_list = self.rearrange_neurons(
            mouse, training_and_test_sessions, data_type="S_binary"
        )
        fps = 15

        running = [
            self.data[mouse][session].spatial.data["running"]
            for session in training_and_test_sessions
        ]

        # Separate neural data into training and test.
        predictor_data = {
            train_test_label: S[:, isrunning].T
            for S, train_test_label, isrunning in zip(
                S_list, ["train", "test"], running
            )
        }
        # predictor_data = {
        #     train_test_label: bin_transients(S, time_bin_size, fps=fps).T
        #     for S, train_test_label in zip(S_list, ["train", "test"])
        # }

        # Separate spatially binned location into training and test.
        outcome_data = dict()
        for session, train_test_label, isrunning in zip(
            sessions, ["train", "test"], running
        ):
            lin_position = session.behavior.data["df"]["lin_position"].values[isrunning]
            outcome_data[train_test_label] = format_spatial_location_for_decoder(
                lin_position,
                n_spatial_bins=n_spatial_bins,
                time_bin_size=1 / fps,
                fps=fps,
                classifier=classifier,
            )

        # Fit the classifier and test on test data.
        classifier.fit(predictor_data["train"], outcome_data["train"])
        y_predicted = classifier.predict(predictor_data["test"])

        # Plot real and predicted spatial location.
        if show_plot:
            fig, ax = plt.subplots()
            ax.plot(outcome_data["test"], alpha=0.5)
            ax.plot(y_predicted, alpha=0.5)

        return y_predicted, predictor_data, outcome_data, classifier

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
        error_time_bin_size=300,
    ):
        goals_to_goals_decoding_error = self.find_decoding_error(
            mouse,
            ("Goals3", "Goals4"),
            classifier=classifier,
            n_spatial_bins=n_spatial_bins,
            error_time_bin_size=error_time_bin_size,
            show_plot=False,
        )

        goals_to_reversal_decoding_error = self.find_decoding_error(
            mouse,
            ("Goals4", "Reversal"),
            classifier=classifier,
            n_spatial_bins=n_spatial_bins,
            error_time_bin_size=error_time_bin_size,
            show_plot=False,
        )

        time_bins_goals = range(len(goals_to_goals_decoding_error))
        time_bins_reversal = (
            np.arange(len(goals_to_reversal_decoding_error)) + time_bins_goals[-1] + 1
        )

        fig, ax = plt.subplots()
        for errors, x in zip(
            [goals_to_goals_decoding_error, goals_to_reversal_decoding_error],
            [time_bins_goals, time_bins_reversal],
        ):
            means = [np.mean(error) for error in errors]
            sems = [np.std(error) / np.sqrt(error.shape[0]) for error in errors]
            ax.errorbar(x, means, yerr=sems)
        ax.set_ylabel("Decoding error [spatial bins]")
        ax.set_xlabel("Time bins")

    def find_decoding_error(
        self,
        mouse,
        training_and_test_sessions,
        classifier=BernoulliNB(),
        n_spatial_bins=36,
        show_plot=True,
        error_time_bin_size=300,
    ):
        """
        Find decoding error between predicted and real spatially
        binned location.

        :parameters
        ---
        See decode_place().
        """
        y_predicted, predictor_data, outcome_data, classifier = self.decode_place(
            mouse,
            training_and_test_sessions,
            classifier=classifier,
            n_spatial_bins=n_spatial_bins,
            show_plot=False,
        )

        d = get_circular_error(
            y_predicted, outcome_data["test"], n_spatial_bins=n_spatial_bins
        )

        bins = make_bins(d, error_time_bin_size, axis=0)
        binned_d = np.split(d, bins)

        if show_plot:
            fig, ax = plt.subplots()
            time_bins = range(len(binned_d))
            mean_errors = [np.mean(dist) for dist in binned_d]
            sem_errors = [np.std(dist) / np.sqrt(dist.shape[0]) for dist in binned_d]
            ax.errorbar(time_bins, mean_errors, yerr=sem_errors)

        return binned_d

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

    def find_assembly_trends(
        self, mouse, session_type, x="time", z_threshold=2.58, x_bin_size=60
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
        assemblies = session.assemblies

        # z-score all assembly activations within assembly.
        z_activations = zscore(assemblies["activations"], axis=1)

        # Binarize activations so that there are no strings of 1s spanning multiple frames.
        binarized_activations = np.zeros_like(z_activations)
        for i, assembly in enumerate(z_activations):
            on_frames = contiguous_regions(assembly > z_threshold)[:, 0]
            binarized_activations[i, on_frames] = 1

        # If binning by time, sum activations every few seconds or minutes.
        if x == "time":
            binned_activations = []

            for assembly in binarized_activations:
                binned_activations.append(
                    bin_transients(assembly[np.newaxis], x_bin_size, fps=15)[0]
                )

            binned_activations = np.vstack(binned_activations)

        # If binning by trials, sum activations every n trials.
        elif x == "trial":
            trial_bins = np.arange(0, session.behavior.data["ntrials"], x_bin_size)
            df = session.behavior.data["df"]

            binned_activations = np.zeros(
                (binarized_activations.shape[0], len(trial_bins))
            )
            for i, assembly in enumerate(binarized_activations):
                for j, (lower, upper) in enumerate(zip(trial_bins, trial_bins[1:])):
                    in_trial = (df["trials"] >= lower) & (df["trials"] < upper)
                    binned_activations[i, j] = np.sum(assembly[in_trial])

        else:
            raise ValueError("Invalid value for x.")

        # Group assemblies into either increasing, decreasing, or no trend in occurrence rate.
        assembly_trends = {
            "no trend": [],
            "decreasing": [],
            "increasing": [],
        }
        for i, assembly in enumerate(binned_activations):
            mk_test = mk.original_test(assembly)
            assembly_trends[mk_test.trend].append(i)

        return assembly_trends, binned_activations

    def plot_assembly_trends(
        self, x="time", x_bin_size=60, z_threshold=2.58, show_plot=True
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
                assembly_trends = self.find_assembly_trends(
                    mouse,
                    session_type,
                    x=x,
                    x_bin_size=x_bin_size,
                    z_threshold=z_threshold,
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

            for c, age in zip(["r", "k"], ["aged", "young"]):
                ax.plot(
                    session_labels,
                    p_decreasing.sel(mouse=self.meta["grouped_mice"][age]).T,
                    color=c,
                )
            ax.set_ylabel("Proportion ensembles with decreasing activity over session")
            plt.setp(ax.get_xticklabels(), rotation=45)

        return assembly_trends, assembly_counts

    def plot_proportion_changing_ensembles(
        self, x="time", x_bin_size=60, z_threshold=2.58, sessions=("Goals4", "Reversal"), trend='decreasing',
    ):
        ensemble_trends, ensemble_counts = self.plot_assembly_trends(
            x=x, x_bin_size=x_bin_size, z_threshold=z_threshold, show_plot=False
        )

        fig, axs = plt.subplots(1, 2, figsize=(7, 6), sharey=True)
        fig.subplots_adjust(wspace=0)
        p_changing = ensemble_counts.sel(trend=trend) / ensemble_counts.sum(
            dim="trend"
        )

        p_changing_split_by_age = dict()
        for ax, age, color in zip(axs, ["young", "aged"], ["w", "r"]):
            p_changing_split_by_age[age] = [
                p_changing.sel(session=session, mouse=self.meta["grouped_mice"][age])
                for session in sessions
            ]
            boxes = ax.boxplot(
                p_changing_split_by_age[age], patch_artist=True, widths=0.75
            )
            for patch, med in zip(boxes["boxes"], boxes["medians"]):
                patch.set_facecolor(color)
                med.set(color="k")

            if age == "aged":
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("Proportion " + trend + " ensembles")
            ax.set_title(age)
            ax.set_xticklabels(sessions, rotation=45)

        return p_changing_split_by_age

    def plot_assembly_by_trend(
        self, mouse, session_type, trend, x="time", x_bin_size=60
    ):
        assembly_trends, binned_activations = self.find_assembly_trends(
            mouse, session_type, x=x, x_bin_size=x_bin_size
        )

        session = self.data[mouse][session_type]
        for assembly_number in assembly_trends[trend]:
            session.plot_assembly(assembly_number)

    def plot_licks(self, mouse, session_type):
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
        ax = self.data[mouse][session_type].behavior.get_licks(plot=True)[1]
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
        self, window=8, strides=2, ax=None, performance_metric="d_prime", show_plot=True
    ):
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
                    rolling_window=window, trial_interval=strides, plot=False
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
            for age in ["aged", "young"]
        }
        metrics = {key: nan_array(dims[key]) for key in ["aged", "young"]}

        for age in ["aged", "young"]:
            for row, mouse in enumerate(self.meta["grouped_mice"][age]):
                for border, session in zip(borders, self.meta["session_types"]):
                    metric_this_session = behavioral_performance.sel(
                        metric=performance_metric, mouse=mouse, session=session
                    ).values.tolist()
                    length = len(metric_this_session)
                    metrics[age][row, border : border + length] = metric_this_session

        if show_plot:
            if ax is None:
                fig, ax = plt.subplots()
            for age, c in zip(["young", "aged"], ["k", "r"]):
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
            ax.set_ylabel(performance_metric)
            ax.set_xlabel("Trial blocks")
            ax = beautify_ax(ax)
            fig.legend()

        return behavioral_performance, metrics

    def plot_best_performance(self, session_type, ax=None, window=8):
        behavioral_performance = self.plot_all_behavior(show_plot=False, window=window)[
            0
        ]

        best_d = dict()
        for age in ["young", "aged"]:
            best_d[age] = []
            for mouse in self.meta["grouped_mice"][age]:
                best_d[age].append(
                    np.nanmax(
                        behavioral_performance.sel(
                            mouse=mouse, metric="d_prime", session=session_type
                        ).values.tolist()
                    )
                )

        label_axes = True
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 4.75))
        else:
            label_axes = False
        box = ax.boxplot(
            [best_d[age] for age in ["young", "aged"]],
            labels=["young", "aged"],
            patch_artist=True,
            widths=0.75,
        )
        for patch, med, color in zip(box["boxes"], box["medians"], ["w", "r"]):
            patch.set_facecolor(color)
            med.set(color="k")

        if label_axes:
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Young", "Aged"])
            ax.set_ylabel("Best d'")
            ax = beautify_ax(ax)
            plt.tight_layout()

        return best_d

    def plot_best_performance_all_sessions(self, window=None):
        fig, axs = plt.subplots(1, len(self.meta["session_types"]), sharey=True)
        fig.subplots_adjust(wspace=0)
        for ax, session in zip(axs, self.meta["session_types"]):
            self.plot_best_performance(session_type=session, ax=ax, window=window)
            ax.set_xticks([])
            ax.set_title(f"{session}")
            if session == "Goals1":
                ax.set_ylabel("d' (per mouse)")

        patches = [
            mpatches.Patch(color=c, label=label)
            for c, label in zip(["w", "r"], ["Young", "Aged"])
        ]
        fig.legend(handles=patches, loc="lower right")

    def match_ensembles(self, mouse, session_types: tuple):
        """
        Match assemblies across two sessions. For each assembly in the first session of the session_types tuple,
        find the corresponding assembly in the second session by taking the highest cosine similarity between two
        assembly patterns.

        :parameters
        ---
        mouse: str
            Mouse name.

        session_types: tuple
            Two session names (e.g. (Goals1, Goals2))

        absolute_value: boolean
            Whether to take the absolute value of the pattern similarity matrix. Otherwise, try negating the pattern
            and take the larger value of the two resulting cosine similarities.

        :returns
        ---
        similarities: {assembly_pair: cosine similarity} dict
            Cosine similarities for every assembly combination.

        """
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
        z_similarities = zscore(
            np.max(similarities, axis=0), axis=0
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
            "poor_matches": ~np.any(z_similarities > 2.58, axis=1),
        }

        return registered_ensembles

    def plot_matched_assemblies(self, mouse, session_types: tuple):
        registered_patterns = self.match_ensembles(mouse, session_types)
        matched_patterns = registered_patterns["matched_patterns"]
        registered_spike_times = self.rearrange_neurons(
            mouse, session_types, "spike_times", detected="everyday"
        )

        for (
            s1_activation,
            s2_activation,
            s1_pattern,
            s2_pattern,
            poor_match,
            similarity,
        ) in zip(
            registered_patterns["matched_activations"][0],
            registered_patterns["matched_activations"][1],
            registered_patterns["matched_patterns"][0],
            registered_patterns["matched_patterns"][1],
            registered_patterns["poor_matches"],
            registered_patterns["best_similarities"],
        ):
            order = np.argsort(np.abs(s1_pattern))

            fig = plt.figure(figsize=(19.2, 10.7))
            spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
            assembly_axs = [fig.add_subplot(spec[i, 0]) for i in range(2)]
            pattern_ax = fig.add_subplot(spec[:, 1])

            for ax, activation, spike_times, pattern, c, session in zip(
                assembly_axs,
                [s1_activation, s2_activation],
                registered_spike_times,
                [s1_pattern, s2_pattern],
                ["k", "r"],
                session_types,
            ):
                # Plot assembly activation.
                plot_assembly(
                    0,
                    activation,
                    spike_times,
                    sort_by_contribution=False,
                    order=order,
                    ax=ax,
                )
                ax.set_title(session)

                # Plot the patterns.
                markerline, stemlines, baseline = pattern_ax.stem(
                    range(len(pattern)), pattern, c, markerfmt=c + "o", basefmt=" "
                )
                plt.setp(markerline, alpha=0.5)
                plt.setp(stemlines, alpha=0.5)

            # Label the patterns.
            pattern_ax.set_ylabel("Weight [a.u.]")
            pattern_ax.set_xlabel("Neurons")
            pattern_ax.legend(session_types)
            title = f"Cosine similarity: {np.round(similarity, 3)}"
            if poor_match:
                title += " NON-SIG MATCH!"
            pattern_ax.set_title(title)
            pattern_ax.set_xticks([0, matched_patterns[0].shape[0]])
            pattern_ax = beautify_ax(pattern_ax)
            plt.gcf().tight_layout()

    def spiralplot_matched_assemblies(self, mouse, session_types: tuple, thresh=1):
        # Match assemblies.
        registered_ensembles = self.match_ensembles(mouse, session_types)

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
        for s1_assembly, s2_assembly in enumerate(registered_ensembles["matches"]):
            activations = [
                self.data[mouse][session_type].assemblies["activations"][assembly]
                for session_type, assembly in zip(
                    session_types, [s1_assembly, s2_assembly]
                )
            ]

            # Make a figure with 2 subplots, one for each session.
            fig, axs = plt.subplots(2, 1, subplot_kw=dict(polar=True))
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
                ax.set_title(f"Ensemble #{assembly_number} in session {session_type}")

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
            for age, color, ax in zip(["young", "aged"], ["w", "r"], axs):
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

    def percent_matched_ensembles(self, session_pair1, session_pair2):
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

        for age, ax, color in zip(["young", "aged"], axs, ["w", "r"]):
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

    def plot_pattern_cosine_similarity_comparisons(self, session_pair1, session_pair2):
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

        for age, ax, color in zip(["young", "aged"], axs, ["w", "r"]):
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
                    best_similarities_this_pair = self.match_ensembles(
                        mouse, (s1, s2)
                    )["best_similarities"]
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

    def make_ensemble_fields(self, mouse, session_type, spatial_bin_size_radians=0.05):
        """
Make a single Pastalkova (snake) plot depicting z-scored assembly activation strength as a function of spatial
location.

:parameters
---
mouse: str
    Mouse name.

session_type: str
    Session name (Goals1, 2, 3, 4, or Reversal).

spatial_bin_size_radians: float
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
        lin_position = behavior_data["df"]["lin_position"]
        placefield_data = self.data[mouse][session_type].spatial

        # Make sure spatial_bin_size_radians matches the one run with PlaceFields().
        if spatial_bin_size_radians is None:
            spatial_bin_size_radians = placefield_data.meta["bin_size"]
        else:
            assert (
                    spatial_bin_size_radians == placefield_data.meta["bin_size"]
            ), "Currently only supports spatial_bin_size_radians that is the same value as the one run with PlaceFields()."

        ensemble_fields = spatial_bin_ensemble_activations(ensembles['activations'], lin_position,
                                                           placefield_data.data['occupancy_map'],
                                                           spatial_bin_size_radians=spatial_bin_size_radians,
                                                           do_zscore=True)
        # Find the spatial bin where activity peaks.
        peak_bins = np.argmax(ensemble_fields, axis=1)

        return ensemble_fields, peak_bins

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
        ensemble_fields, peak_bins = self.make_ensemble_fields(mouse, session_type)
        behavior_data = self.data[mouse][session_type].behavior.data
        lin_position = behavior_data['df']['lin_position']
        port_locations = np.asarray(behavior_data['lin_ports'])[behavior_data['rewarded_ports']]

        # Determine order of assemblies.
        if order is None and do_sort:
            order = np.argsort(peak_bins)
        elif order is None and not do_sort:
            order = range(ensemble_fields.shape[0])

        # Plot assemblies.
        if show_plot:
            # Convert port locations to bin #.
            bins = spatial_bin(
                lin_position,
                np.zeros_like(lin_position),
                bin_size_cm=spatial_bin_size_radians,
                show_plot=False,
                one_dim=True,
            )[-1]
            reward_locations_bins = np.digitize(port_locations, bins) - 1

            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5.5))
            ax.imshow(ensemble_fields[order])
            ax.axis("tight")
            ax.set_ylabel("Ensemble #")
            ax.set_xlabel("Location")

            for port in reward_locations_bins:
                ax.axvline(port, c="g")

        return ensemble_fields

    def map_ensemble_fields(self, mouse, session_types, spatial_bin_size_radians=None):
        # Register the ensembles.
        registered_ensembles = self.match_ensembles(mouse, session_types)

        # Get linearized positions, occupancies, port locations, spatial bin size in radians.
        lin_positions = [self.data[mouse][session].behavior.data['df']['lin_position'] for session in session_types]
        occupancies = [self.data[mouse][session].spatial.data['occupancy_map'] for session in session_types]
        if spatial_bin_size_radians is None:
            spatial_bin_size_radians = [self.data[mouse][session].spatial.meta['bin_size'] for session in session_types]
            assert len(np.unique(spatial_bin_size_radians))==1, 'Different bin sizes in two sessions.'
            spatial_bin_size_radians = spatial_bin_size_radians[0]

        # Make the fields and omit the poor matches.
        ensemble_fields = [
            spatial_bin_ensemble_activations(activations, positions, occupancy,
                                             spatial_bin_size_radians=spatial_bin_size_radians)
            for activations, positions, occupancy in zip(registered_ensembles['matched_activations'],
                                                         lin_positions,
                                                         occupancies)]
        ensemble_fields[1][registered_ensembles['poor_matches']] = np.nan

        return ensemble_fields

    def correlate_ensemble_fields(self, mouse, session_types, spatial_bin_size_radians=None):
        ensemble_fields = self.map_ensemble_fields(mouse, session_types, spatial_bin_size_radians=spatial_bin_size_radians)

        rhos = [spearmanr(ensemble_day1, ensemble_day2)[0]
                for ensemble_day1, ensemble_day2 in zip(ensemble_fields[0], ensemble_fields[1])]

        return np.asarray(rhos)

    def plot_ensemble_field_correlations(self, session_pair:tuple, show_plot=True):
        ensemble_field_rhos = dict()
        for age in ['young', 'aged']:
            ensemble_field_rhos[age] = []

            for mouse in self.meta['grouped_mice'][age]:
                rhos = self.correlate_ensemble_fields(mouse, session_pair)
                ensemble_field_rhos[age].append(rhos[~np.isnan(rhos)])

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, color, ax in zip(['young', 'aged'], ['w', 'r'], axs):
                boxes = ax.boxplot(ensemble_field_rhos[age], patch_artist=True)

                for patch, med in zip(boxes['boxes'], boxes['medians']):
                    patch.set_facecolor(color)
                    med.set(color='k')

                if age == 'aged':
                    ax.set_yticks([])
                else:
                    ax.set_ylabel('Spearman rho of matched ensemble fields')
                ax.set_xticklabels(self.meta['grouped_mice'][age], rotation=45)

        return ensemble_field_rhos


    def plot_ensemble_field_correlation_comparisons(self, session_pair1:tuple, session_pair2:tuple, show_plot=True):
        ensemble_field_rhos = dict()
        for session_pair in [session_pair1, session_pair2]:
            ensemble_field_rhos[session_pair] = self.plot_ensemble_field_correlations(session_pair, show_plot=False)

        if show_plot:
            fig, axs = plt.subplots(1, 2)
            fig.subplots_adjust(wspace=0)

            for age, ax, color in zip(['young', 'aged'], axs, ['w', 'r']):
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
                    ax.set_ylabel("Cosine similarity of matched assemblies")
                ax.set_xticks(label_positions)
                ax.set_xticklabels(mice, rotation=45)

        return ensemble_field_rhos

    #def ensemble_

    def snakeplot_matched_ensembles(self, mouse, session_types, spatial_bin_size_radians=None,
                                    show_plot=True, axs=None, sort_on=0):
        # Map the ensembles to each other.
        ensemble_fields = self.map_ensemble_fields(mouse, session_types,
                                                   spatial_bin_size_radians=spatial_bin_size_radians)

        # Get linearized position and port locations.
        lin_positions = [self.data[mouse][session].behavior.data['df']['lin_position'] for session in session_types]
        port_locations = [np.asarray(self.data[mouse][session].behavior.data["lin_ports"])
                          [self.data[mouse][session].behavior.data["rewarded_ports"]] for session in session_types]

        if spatial_bin_size_radians is None:
            spatial_bin_size_radians = [self.data[mouse][session].spatial.meta['bin_size'] for session in session_types]
            assert len(np.unique(spatial_bin_size_radians))==1, 'Different bin sizes in two sessions.'
            spatial_bin_size_radians = spatial_bin_size_radians[0]

        # Convert port locations to bin #.
        bins = [spatial_bin(
            position,
            np.zeros_like(position),
            bin_size_cm=spatial_bin_size_radians,
            show_plot=False,
            one_dim=True,
        )[-1] for position in lin_positions]
        port_locations_bins = [np.where(
            spatial_bin(
                port,
                np.zeros_like(port),
                bin_size_cm=spatial_bin_size_radians,
                show_plot=False,
                one_dim=True,
                bins=bins_,
            )[0]
        )[0] for port, bins_ in zip(port_locations, bins)]

        # Sort the fields.
        order = np.argsort(np.argmax(ensemble_fields[sort_on], axis=1))

        # Plot.
        if show_plot:
            if axs is None:
                fig, axs = plt.subplots(1, len(session_types))

            for ax, fields, ports in zip(axs, ensemble_fields, port_locations_bins):
                ax.imshow(fields[order])
                ax.axis('tight')
                ax.set_ylabel('Ensemble #')
                ax.set_xlabel('Location')

                for port in ports:
                    ax.axvline(port, c='r', alpha=0.5)

        return ensemble_fields

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
    # mice = [
    #     # "Betelgeuse_Scope25",
    #     # "Alcor_Scope20",
    #     "Castor_Scope05",
    #     # "Draco_Scope02",
    #     "Encedalus_Scope14",
    #     "Fornax",
    #     "Hydra",
    #     "Io",
    #     #"Janus",
    #     "Kalyke",
    #     "Lyra",
    #     # "M1",
    #     # "M2",
    #     # "M3",
    #     # "M4",
    # ]
    # # B = BatchBehaviorAnalyses(mice)
    # # B.plot_learning_trials_per_mouse()
    # # B.plot_all_session_licks()
    # # B.plot_all_sdts(1)
    # # B.compare_d_prime(8, 'CircleTrackReversal1', 'CircleTrackReversal2')
    #
    # B = ProjectAnalyses(mice)
    # B.correlate_stability_to_reversal()
    # #B.spiral_scrollplot_assemblies('Castor_Scope05', 'CircleTrackReversal1')
    # lapsed_assemblies, spiking = B.plot_lapsed_assemblies('Castor_Scope05', ('CircleTrackGoals2','CircleTrackReversal1'))

    RR = ProjectAnalyses(
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
        ],
        project_name="RemoteReversal",
        behavior_only=False,
    )
    RR.data["Fornax"]["Goals3"].sdt_trials()
    pass
