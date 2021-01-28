from CircleTrack.BehaviorFunctions import BehaviorSession
from CircleTrack.MiniscopeFunctions import CalciumSession
import matplotlib.pyplot as plt
from CaImaging.util import (
    sem,
    nan_array,
    bin_transients,
    make_bins,
    ScrollPlot,
)
from CaImaging.plotting import errorfill
from grid_strategy.strategies import RectangularStrategy, SquareStrategy
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches
from CircleTrack.SessionCollation import MultiAnimal
from CaImaging.CellReg import rearrange_neurons, trim_map, scrollplot_footprints
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LinearRegression
from CaImaging.Assemblies import lapsed_activation, plot_assemblies, membership_sort
from scipy.stats import circmean
import numpy as np
from scipy.stats import mode, zscore
import os
from CircleTrack.plotting import plot_daily_rasters, plot_spiral
from CaImaging.Assemblies import preprocess_multiple_sessions, lapsed_activation
from CircleTrack.Assemblies import plot_assembly
from itertools import product

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 12})


class BatchFullAnalyses:
    def __init__(self, mice, project_folder=r"Z:\Will\Drift\Data"):
        # Collect data from all mice and sessions.
        self.data = MultiAnimal(
            mice, project_folder, behavior="CircleTrack", SessionFunction=CalciumSession
        )

        # Define session types here. Watch out for typos.
        # Order matters. Plots will be in the order presented here.
        self.meta = {
            "session_types": [
                "CircleTrackShaping1",
                "CircleTrackShaping2",
                "CircleTrackGoals1",
                "CircleTrackGoals2",
                "CircleTrackReversal1",
                "CircleTrackReversal2",
                "CircleTrackRecall",
            ],
            "mice": mice,
        }

        self.meta["session_labels"] = [
            session_type.replace("CircleTrack", "")
            for session_type in self.meta["session_types"]
        ]

        # for mouse in mice:
        #     S_list = [self.data[mouse][session].data['imaging']['S']
        #               for session in self.meta['session_types]]
        #
        #     cell_map = self.data[mouse]['CellReg'].map
        #
        #     rearranged = rearrange_neurons(cell_map, S_list)

    def count_ensembles(self):
        n_ensembles = nan_array((len(self.meta['mice']), len(self.meta['session_types'])))
        for i, mouse in enumerate(self.meta['mice']):
            for j, session_type in enumerate(self.meta['session_types']):
                session = self.data[mouse][session_type]
                n_ensembles[i,j] = session.assemblies['significance'].nassemblies

        fig, ax = plt.subplots()
        for n, mouse in zip(n_ensembles, self.meta['mice']):
            ax.plot(self.meta['session_labels'], n)
            ax.annotate(mouse, (0.1, n[0] + 1))

        ax.set_ylabel('# of ensembles')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        fig.subplots_adjust(bottom=0.2)

    def rearrange_neurons(self, mouse, session_types, data_type, detected='everyday'):
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
        n_sessions = len(self.meta['session_types'])
        overlaps = nan_array((len(self.meta['mice']), n_sessions, n_sessions))

        for i, mouse in enumerate(self.meta['mice']):
            overlaps[i] = self.find_percent_overlap(mouse, show_plot=False)

        if show_plot:
            fig, ax = plt.subplots()
            m = np.mean(overlaps, axis=0)
            se = sem(overlaps, axis=0)
            x = self.meta['session_labels']
            for y, yerr in zip(m, se):
                errorfill(x, y, yerr, ax=ax)

            ax.set_ylabel('Proportion of registered cells')
            plt.setp(ax.get_xticklabels(), rotation=45)

        return overlaps, ax

    def find_percent_overlap(self, mouse, show_plot=True):
        n_sessions = len(self.meta['session_types'])
        r, c = np.indices((n_sessions, n_sessions))
        overlaps = np.ones((n_sessions, n_sessions))
        for session_pair, i, j in zip(
                product(self.meta['session_types'], self.meta['session_types']),
                r.flatten(), c.flatten()):
            if i != j:
                overlap_map = self.get_cellreg_mappings(mouse,
                                                        session_pair,
                                                        detected='first_day')[0]

                n_neurons_reactivated = np.sum(overlap_map.iloc[:,1] != -9999)
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
        goals = ("CircleTrackGoals1", "CircleTrackGoals2")
        reversal = ("CircleTrackGoals2", "CircleTrackReversal1")

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

    def correlate_stability_to_reversal(
        self,
        corr_sessions=("CircleTrackGoals1", "CircleTrackGoals2"),
        criterion_session="CircleTrackReversal1",
    ):
        median_r = []
        criterion = []
        for mouse in self.meta["mice"]:
            corrs = self.correlate_fields(mouse, corr_sessions, show_histogram=False)
            median_r.append(np.nanmedian(corrs["r"]))
            behavior = self.data[mouse][criterion_session].behavior
            criterion.append(behavior.data["learning"]["criterion"])

        fig, ax = plt.subplots()
        ax.scatter(median_r, criterion)
        ax.set_xlabel("Median place field correlation Goals1 vs Goals 2 [r]")
        ax.set_ylabel("Trials to criterion Reversal1")

        return median_r, criterion

    def get_placefield_distribution_comparisons(
        self,
        session_pair1=("CircleTrackGoals1", "CircleTrackGoals2"),
        session_pair2=("CircleTrackGoals2", "CircleTrackReversal1"),
    ):
        r = {
            "pair1": [],
            "pair2": [],
        }
        for mouse in self.meta["mice"]:
            for key, pair in zip(["pair1", "pair2"], [session_pair1, session_pair2]):
                r[key].append(
                    self.correlate_fields(mouse, pair, show_histogram=False)["r"]
                )

        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 5)
        ax.boxplot([np.asarray(data)[~np.isnan(data)] for data in
                    r['pair1']], positions=x)

        x = np.linspace(2, 3, 5)
        ax.boxplot([np.asarray(data)[~np.isnan(data)] for data in
                    r['pair2']], positions=x)
        ax.set_ylabel('Firing field correlations [r]')

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
        time_bin_size=1,
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

        # Separate neural data into training and test.
        predictor_data = {
            train_test_label: bin_transients(S, time_bin_size, fps=fps).T
            for S, train_test_label in zip(S_list, ["train", "test"])
        }

        # Separate spatially binned location into training and test.
        outcome_data = dict()
        for session, train_test_label in zip(sessions, ["train", "test"]):
            lin_position = session.behavior.data["df"]["lin_position"].values
            outcome_data[train_test_label] = self.format_spatial_location_for_decoder(
                lin_position,
                n_spatial_bins=n_spatial_bins,
                time_bin_size=time_bin_size,
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
        decoder_time_bin_size=1,
        n_spatial_bins=36,
        error_time_bin_size=300,
    ):
        goals_to_goals_decoding_error = self.find_decoding_error(
            mouse,
            ("CircleTrackGoals1", "CircleTrackGoals2"),
            classifier=classifier,
            decoder_time_bin_size=decoder_time_bin_size,
            n_spatial_bins=n_spatial_bins,
            error_time_bin_size=error_time_bin_size,
            show_plot=False,
        )

        goals_to_reversal_decoding_error = self.find_decoding_error(
            mouse,
            ("CircleTrackGoals2", "CircleTrackReversal1"),
            classifier=classifier,
            decoder_time_bin_size=decoder_time_bin_size,
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
        decoder_time_bin_size=1,
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
            time_bin_size=decoder_time_bin_size,
            n_spatial_bins=n_spatial_bins,
            show_plot=False,
        )

        d = self.get_circular_error(
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

    def get_circular_error(self, y_predicted, y_real, n_spatial_bins):
        """
        Error is not linear here, it's circular because for example,
        spatial bin 1 is right next to spatial bin 36. Take the
        circular distance here.

        :parameters
        ---
        y_predicted: array-like of ints
            Predicted spatially binned locations.

        y_real: array-like of ints
            Real spatially binned locations.

        n_spatial_bins: int
            Number of spatial bins (needed for modulo function).
        """
        i = (y_predicted - y_real) % n_spatial_bins
        j = (y_real - y_predicted) % n_spatial_bins

        d = np.min(np.vstack((i, j)), axis=0)

        return d

    def format_spatial_location_for_decoder(
        self,
        lin_position,
        n_spatial_bins=36,
        time_bin_size=1,
        fps=15,
        classifier=BernoulliNB(),
    ):
        """
        Naive Bayes classifiers only take integers as outcomes.
        Bin spatial locations both spatially and temporally.

        :parameters
        ---
        lin_position: array-like of floats
            Linearized position (in radians, from behavioral DataFrame).

        n_spatial_bins: int
            Number of spatial bins.

        decoder_time_bin_size: float
            Size of temporal bin in seconds.

        fps: int
            Frames per second of the acquired data.
        """
        # Find the appropriate bin edges given number of spatial bins.
        # Then do spatial bin.
        dont_bin_space = True if isinstance(classifier, (LinearRegression)) else False
        if dont_bin_space:
            binned_position = np.cos(lin_position)
        else:
            bins = np.histogram(lin_position, bins=n_spatial_bins)[1]
            binned_position = np.digitize(lin_position, bins)

        # Do the same for temporal binning.
        bins = make_bins(binned_position, fps * time_bin_size, axis=0)
        binned_position = np.split(binned_position, bins, axis=0)

        # Get the most occupied spatial bin within each temporal bin.
        if dont_bin_space:
            position = np.array([circmean(time_bin) for time_bin in binned_position])
        else:
            position = np.array([mode(time_bin)[0][0] for time_bin in binned_position])

        return position

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
        self, mouse, session_types, smooth_factor=5, use_bool=True, z_method="global", detected='everyday'
    ):
        S_list = self.rearrange_neurons(mouse, session_types, "S", detected=detected)
        data = preprocess_multiple_sessions(
            S_list, smooth_factor=smooth_factor, use_bool=use_bool, z_method=z_method
        )

        lapsed_assemblies = lapsed_activation(data["processed"])

        return lapsed_assemblies

    def plot_lapsed_assemblies(self, mouse, session_types, detected='everyday'):
        lapsed_assemblies = self.get_lapsed_assembly_activation(mouse, session_types, detected=detected)
        spiking = self.rearrange_neurons(mouse, session_types, "spike_times", detected=detected)

        n_sessions = len(lapsed_assemblies['activations'])
        for i, pattern in enumerate(lapsed_assemblies['patterns']):
            fig, axs = plt.subplots(n_sessions, 1)
            for ax, activation, spike_times in zip(axs,
                                                   lapsed_assemblies[
                                                       'activations'],
                                                   spiking):
                plot_assembly(pattern, activation[i], spike_times, ax=ax)

        return lapsed_assemblies, spiking


class BatchBehaviorAnalyses:
    def __init__(self, mice, project_folder=r"Z:\Will\Drift\Data"):
        """
        This class definition will contain behavior analyses spanning
        the entire dataset (or at least the specified mice).

        :param mice:
        :param project_folder:
        """
        # Compile data for all animals.
        self.data = dict()
        self.data["mice"] = MultiAnimal(
            mice,
            project_folder,
            behavior="CircleTrack",
            SessionFunction=BehaviorSession,
        )
        self.meta = dict()
        self.meta["mice"] = mice
        self.meta["n_mice"] = len(mice)

        # Define session types here. Watch out for typos.
        # Order matters. Plots will be in the order presented here.
        self.meta["session_types"] = [
            "CircleTrackShaping1",
            "CircleTrackShaping2",
            "CircleTrackGoals1",
            "CircleTrackGoals2",
            "CircleTrackReversal1",
            "CircleTrackReversal2",
            "CircleTrackRecall",
        ]

        # Same as session types, just without 'CircleTrack'.
        self.meta["session_labels"] = [
            session_type.replace("CircleTrack", "")
            for session_type in self.meta["session_types"]
        ]

        # Gather the number of trials for each session type and
        # arrange it in a (mouse, session) array.
        self.data["trial_counts"], self.data["max_trials"] = self.count_trials()
        (
            self.data["licks"],
            self.data["rewarded_ports"],
            self.data["n_drinks"],
            self.data["p_drinks"],
        ) = self.resort_data()
        self.data["learning_trials"] = self.get_learning_trials()

        pass

    def signal_detection_analysis(self, n_trial_blocks):
        """
        Run signal detection on all sessions. This calculates
        hits, misses, correct rejections, and false alarms over
        a series of trials. Specify the number of trials to
        compute signal detection over for each session.
        n_trial_blocks will evenly split trials into that
        number of blocks.

        :parameter
        ---
        n_trial_blocks: int
            Splits all trials in that session into n blocks.
        """
        # Compute signal detection calculations.
        for sessions in self.data["mice"].values():
            for session in sessions.values():
                session.SDT(n_trial_blocks=n_trial_blocks, plot=False)

        # Resort data into session types, listed by mouse.
        sdt_matrix = dict()
        sdt_categories = ["hits", "misses", "FAs", "CRs", "d_prime"]
        for session_type in self.meta["session_types"]:
            sdt_matrix[session_type] = {
                key: nan_array((self.meta["n_mice"], n_trial_blocks))
                for key in sdt_categories
            }

            for m, mouse in enumerate(self.meta["mice"]):
                mouse_data = self.data["mice"][mouse]

                try:
                    for key in sdt_categories:
                        sdt_matrix[session_type][key][m] = mouse_data[session_type].sdt[
                            key
                        ]
                except KeyError:
                    print(
                        f"{session_type} not found for " f"mouse {mouse}! Skipping..."
                    )

        return sdt_matrix

    def plot_all_sdts(self, n_trial_blocks):
        """
        Plot the hit and correct rejection rate for all sessions.
        Also plot d'.

        :parameter
        ---
        n_trial_blocks: int
            Splits all trials in that session into n blocks.

        """
        # Build the legend.
        hit_patch = mpatches.Patch(color="forestgreen", label="Hits")
        cr_patch = mpatches.Patch(color="steelblue", label="Correct rejections")
        dprime_patch = mpatches.Patch(color="k", label="d'")
        # Handle the case where you want the entire session's hit/
        # correct rejection rate/d'. Plots two subplots, one of
        # hit/correct rejection rate, another for d' across all sessions.
        if n_trial_blocks == 1:
            # Preallocate.
            fig, axs = plt.subplots(2, 1, figsize=(7, 9.5))
            hits = nan_array((self.meta["n_mice"], len(self.meta["session_types"])))
            CRs = nan_array((self.meta["n_mice"], len(self.meta["session_types"])))
            d_primes = nan_array((self.meta["n_mice"], len(self.meta["session_types"])))

            # Acquire data and sort by session.
            for s, (session_type, label) in enumerate(
                zip(self.meta["session_types"], self.meta["session_labels"])
            ):
                self.verify_sdt(1, session_type, "hits")
                self.verify_sdt(1, session_type, "CRs")
                self.verify_sdt(1, session_type, "d_prime")

                for m, mouse in enumerate(self.meta["mice"]):
                    try:
                        session_data = self.data["mice"][mouse][session_type]

                        hits[m, s] = session_data.sdt["hits"][0]
                        CRs[m, s] = session_data.sdt["CRs"][0]
                        d_primes[m, s] = session_data.sdt["d_prime"][0]
                    except KeyError:
                        print(f"{session_type} not found for " f"mouse {mouse}!")

            # Plot the values.
            axs[0].plot(
                self.meta["session_labels"],
                hits.T,
                "o-",
                color="forestgreen",
                label="Hits",
            )
            axs[0].set_ylabel("%")
            axs[0].plot(CRs.T, "o-", color="steelblue", label="Correct rejections")
            axs[1].plot(self.meta["session_labels"], d_primes.T, "o-", color="black")
            axs[1].set_ylabel("d'")
            for ax in axs:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()

            axs[0].legend(handles=[hit_patch, cr_patch])

        # Otherwise, split the sessions into trial blocks and
        # plot d' etc per block.
        else:
            # Preallocate the figure axes.
            fig, axs = plt.subplots(4, 2, sharey="all", figsize=(7, 9.5))
            d_prime_axs = []
            for ax, session_type, label in zip(
                axs.flatten(), self.meta["session_types"], self.meta["session_labels"]
            ):
                self.plot_sdt("hits", n_trial_blocks, session_type, ax)

                # Ignore shaping sessions.
                if "Shaping" not in session_type:
                    self.plot_sdt("CRs", n_trial_blocks, session_type, ax)

                    # Use a different axis for d'.
                    d_prime_ax = ax.twinx()
                    d_prime_axs.append(d_prime_ax)
                    self.plot_sdt("d_prime", n_trial_blocks, session_type, d_prime_ax)
                ax.set_title(label)

            # Link the d' axes.
            for d_prime_ax in d_prime_axs[1:]:
                d_prime_axs[0].get_shared_y_axes().join(d_prime_axs[0], d_prime_ax)
            d_prime_axs[0].autoscale()
            fig.tight_layout(pad=0.5)

            axs.flatten()[-1].legend(handles=[hit_patch, cr_patch, dprime_patch])
        pass

    def verify_sdt(self, n_trial_blocks, session_type, category):
        """
        Verify that the current values stored in self.sdt match the
        expected size of the trial blocks. If self.sdt doesn't exist,
        compute it.

        :parameters
        ---
        n_trial_blocks: int
            Splits all trials in that session into n blocks.

        session_type: str
            Training stage (e.g., CircleTrackShaping1).

        category: str
            'hits', 'misses', 'FAs', or 'CRs'.
        """
        if not hasattr(self, "sdt"):
            self.sdt = self.signal_detection_analysis(n_trial_blocks)
        elif self.sdt[session_type][category].shape[1] != n_trial_blocks:
            self.sdt = self.signal_detection_analysis(n_trial_blocks)

    def compare_d_prime(self, n_trial_blocks, session1_type, session2_type):
        """
        Compare the d' across trial blocks of different session types.

        :parameters
        ---
        n_trial_blocks: int
            Splits all trials in that session into n blocks.

        session1_type and session2_type: strs
            Session types that you want to compare d' from.

        """
        # Make sure the d' exists first.
        self.verify_sdt(n_trial_blocks, session1_type, "d_prime")
        self.verify_sdt(n_trial_blocks, session2_type, "d_prime")

        # Get the data from those sesssions and mice.
        session1 = self.sdt[session1_type]["d_prime"]
        session2 = self.sdt[session2_type]["d_prime"]

        # Get the best way to arrange subplots. Plot all d's.
        # Also record p-values of signed-rank tests and
        # get their multiple comparisons corrected values.
        grid = RectangularStrategy.get_grid_arrangement(n_trial_blocks)
        fig, axs = plt.subplots(
            max(grid), len(grid), sharex="all", sharey="all", figsize=(6.5, 7.5)
        )
        flattened_axs = axs.flatten()
        p_vals = []
        for ax, block in zip(flattened_axs, range(session1.shape[1])):
            ax.scatter(session1[:, block], session2[:, block])
            p = wilcoxon(session1[:, block], session2[:, block]).pvalue
            p_vals.append(p)
        corrected_ps = multipletests(p_vals, method="fdr_bh")[1]

        # Get size of y=x line.
        ax = flattened_axs[0]
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # Get x and y axis labels.
        labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session_type)]
            for session_type in [session1_type, session2_type]
        ]

        # Labels.
        for ax, p, cp in zip(flattened_axs, p_vals, corrected_ps):
            ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
            ax.set_xlabel(f"{labels[0]} d'")
            ax.set_ylabel(f"{labels[1]} d'")
            ax.set_title(f"p = {str(np.round(p, 2))}, " f"{str(np.round(cp, 2))}")
        fig.tight_layout()

        return session1, session2

    def plot_sdt(self, category, n_trial_blocks, session_type, ax=None):
        """
        Plot the signal detection accuracy for a given session
        type and split into n blocks.

        :parameters
        ---
        category: str
            'hits', 'misses', 'FAs', 'CRs', or 'd_prime'

        n_trial_blocks: int
            Splits all trials in that session into n blocks.

        session_type: str
            'CircleTrackingGoals1', etc.

        """
        # Get signal detection computations if not already there.
        self.verify_sdt(n_trial_blocks, session_type, category)

        # Color differs for each category.
        colors = {
            "hits": "forestgreen",
            "misses": "darkred",
            "FAs": "goldenrod",
            "CRs": "steelblue",
            "d_prime": "black",
        }

        # For spacing apart data points.
        jitter = {"hits": -0.1, "misses": -0.1, "FAs": 0.1, "CRs": 0.1, "d_prime": 0}

        if ax is None:
            fig, ax = plt.subplots()

        # Get the x and y axis values.
        plot_me = self.sdt[session_type][category].T
        trial_number = np.asarray(list(range(n_trial_blocks)), dtype=float)
        trial_number += jitter[category]

        ax.plot(trial_number, plot_me, "o-", color=colors[category], alpha=0.6)
        ax.set_xlabel("Trial blocks")

        # Only put % on the left side of the plot.
        if "d_prime" not in category:
            ax.set_ylabel("%")

    def get_learning_trials(self):
        learning_trials = {
            "start": nan_array((self.meta["n_mice"], len(self.meta["session_types"]))),
            "inflection": nan_array(
                (self.meta["n_mice"], len(self.meta["session_types"]))
            ),
        }

        for s, session_type in enumerate(self.meta["session_types"]):
            for m, mouse in enumerate(self.meta["mice"]):
                mouse_data = self.data["mice"][mouse]
                try:
                    learning_trials["start"][m, s] = mouse_data[session_type].data[
                        "learning"
                    ]["start"]
                    learning_trials["inflection"][m, s] = mouse_data[session_type].data[
                        "learning"
                    ]["inflection"]
                except KeyError:
                    pass

        return learning_trials

    def plot_learning_trials_across_sessions(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            self.meta["session_labels"],
            self.data["learning_trials"]["inflection"].T,
            "yo-",
        )
        ax.set_ylabel("Learning inflection trial #")

    def plot_learning_trials_per_mouse(self):
        for mouse in self.meta["mice"]:
            mouse_data = self.data["mice"][mouse]
            fig, axs = plt.subplots(3, 2, sharex="all", sharey="all", figsize=(6.4, 6))

            sessions = [
                session
                for session in self.meta["session_types"]
                if "Shaping" not in session
            ]
            labels = [
                label for label in self.meta["session_labels"] if "Shaping" not in label
            ]
            for ax, session, label in zip(axs.flatten(), sessions, labels):
                try:
                    mouse_data[session].plot_learning_curve(ax=ax)
                    ax.set_title(label)
                except KeyError:
                    pass

            fig.tight_layout(pad=0.5)

            start_patch = mpatches.Patch(color="g", label="Start of learning")
            inflection_patch = mpatches.Patch(color="y", label="Inflection point")
            criterion_patch = mpatches.Patch(color="b", label="Criterion")
            axs.flatten()[-1].legend(
                handles=[start_patch, inflection_patch, criterion_patch]
            )
        pass

    def count_trials(self):
        """
        Count the number of trials for each mouse and each session type.

        :return:
        """
        trial_counts = nan_array((self.meta["n_mice"], len(self.meta["session_types"])))
        for s, session_type in enumerate(self.meta["session_types"]):
            for m, mouse in enumerate(self.meta["mice"]):
                mouse_data = self.data["mice"][mouse]
                try:
                    trial_counts[m, s] = int(mouse_data[session_type].data["ntrials"])
                except KeyError:
                    trial_counts[m, s] = np.nan

        # Get the highest number of trials across all mice for a particular
        # session type.
        max_trials = np.nanmax(trial_counts, axis=0).astype(int)

        return trial_counts, max_trials

    def plot_trials(self):
        """
        Plot number of trials for each mouse over the experiment.

        :return:
        """
        fig, ax = plt.subplots()
        ax.plot(self.meta["session_labels"], self.data["trial_counts"].T, "o-")
        ax.set_xlabel("Session type")
        ax.set_ylabel("Number of trials")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()

    def resort_data(self):
        """
        For each mouse, session type, and trial, get the number of
        licks.
        :return:
        """
        # Get lick data and rearrange it so that it's in a dict where
        # every key references a session type. In those entries,
        # make a (mouse, trial, port #) matrix.
        lick_matrix = dict()
        rewarded_matrix = dict()
        drink_matrix = dict()
        p_drink_matrix = dict()
        learning_start = dict()
        learning_inflection = dict()
        for max_trials, session_type in zip(
            self.data["max_trials"], self.meta["session_types"]
        ):
            lick_matrix[session_type] = nan_array((self.meta["n_mice"], max_trials, 8))
            rewarded_matrix[session_type] = np.zeros(
                (self.meta["n_mice"], 8), dtype=bool
            )
            drink_matrix[session_type] = nan_array((self.meta["n_mice"], max_trials))
            p_drink_matrix[session_type] = nan_array((self.meta["n_mice"], max_trials))
            learning_start[session_type] = nan_array((self.meta["n_mice"],))
            learning_inflection[session_type] = nan_array((self.meta["n_mice"],))

            # Get data and sort by session type. .
            for m, mouse in enumerate(self.meta["mice"]):
                mouse_data = self.data["mice"][mouse]
                try:
                    session_licks = mouse_data[session_type].data["all_licks"]
                    mat_size = session_licks.shape
                    lick_matrix[session_type][
                        m, : mat_size[0], : mat_size[1]
                    ] = session_licks

                    # Also get which ports were rewarded.
                    rewarded = mouse_data[session_type].data["rewarded_ports"]
                    rewarded_matrix[session_type][m] = rewarded

                    # Also get number of drinks for each trial.
                    session_drinks = mouse_data[session_type].data["n_drinks"]
                    drink_matrix[session_type][
                        m, : session_drinks.shape[0]
                    ] = session_drinks

                    # If the session is called 'Shaping', mark
                    # all ports as rewarded. Some ports get marked
                    # as non-rewarded sometimes because they were never
                    # visited due mouse shyness (⁄ ⁄•⁄ω⁄•⁄ ⁄)
                    if "Shaping" in session_type and not all(rewarded):
                        print(
                            "Non-rewarded ports found during a "
                            "shaping session. Setting all ports "
                            "to rewarded"
                        )
                        rewarded_matrix[session_type][m] = np.ones_like(
                            rewarded_matrix, dtype=bool
                        )

                    # And percentage of water deliveries out of all rewarded ports.
                    n_rewarded_ports = np.sum(rewarded_matrix[session_type][m])
                    p_drink_matrix[session_type][m, : session_drinks.shape[0]] = (
                        session_drinks / n_rewarded_ports
                    )
                except KeyError:
                    print(f"{session_type} not found for mouse {mouse}! Skipping...")

        return lick_matrix, rewarded_matrix, drink_matrix, p_drink_matrix

    def plot_all_session_licks(self):
        """
        Categorizes ports as ones that are currently rewarded,
        rewarded from the last session, or not recently rewarded.
        Collapses across those categories and plots lick rates
        across all sessions.

        :return:
        """
        # Plot lick rates.
        fig, axs = plt.subplots(4, 2, sharey="all", sharex="all", figsize=(6.6, 9.4))
        for ax, session_type, label in zip(
            axs.flatten(), self.meta["session_types"], self.meta["session_labels"]
        ):
            self.plot_rewarded_licks(session_type, ax)
            ax.set_title(label)

        # Build the legend.
        rewarded_patch = mpatches.Patch(color="cornflowerblue", label="Correct licks")
        prev_rewarded_patch = mpatches.Patch(
            color="lightcoral", label="Perseverative licks"
        )
        not_rewarded_patch = mpatches.Patch(color="gray", label="Error licks")
        axs.flatten()[-1].legend(
            handles=[rewarded_patch, prev_rewarded_patch, not_rewarded_patch]
        )

        fig.tight_layout(pad=0.2)

        pass

    def plot_rewarded_licks(self, session_type, ax=None):
        """
        Plot the licks rewarded and non-rewarded port averaged
        across animals for each trial.

        :param session_type:
        :return:
        """
        # Get the licks across mice for that session.
        # Also get the port numbers that were rewarded.
        licks = self.data["licks"][session_type]
        rewarded_ports = self.data["rewarded_ports"][session_type]

        # Find the previous session to figure out what was previously rewarded.
        try:
            previously_rewarded = self.get_previous_rewards(session_type)
        except AssertionError:
            previously_rewarded = np.ones_like(rewarded_ports, dtype=bool)

        # If the rewards from the last session match this session,
        # we're going to treat this a little differently.
        same_rewards = True if np.all(previously_rewarded == rewarded_ports) else False
        if np.any(previously_rewarded == rewarded_ports) and not same_rewards:
            print(
                f"Warning! At least one reward port for {session_type} overlaps "
                "with the previous day. Are you sure this is correct?"
            )

        # Some mice might not have the specified session.
        # Exclude those mice.
        mice_to_include = [
            session_type in mouse for mouse in self.data["mice"].values()
        ]
        n_mice = np.sum(mice_to_include)
        licks = licks[mice_to_include]
        rewarded_ports = rewarded_ports[mice_to_include]
        previously_rewarded = previously_rewarded[mice_to_include]

        # Find the number of rewarded ports to allocate
        # two arrays -- one each for rewarded, previously rewarded,
        # and non-rewarded licks.
        n_rewarded = np.unique(np.sum(rewarded_ports, axis=1))
        n_previously_rewarded = np.unique(np.sum(previously_rewarded, axis=1))
        assert len(n_rewarded) == 1, "Number of rewarded ports differ in some mice!"
        assert (
            len(n_previously_rewarded) == 1
        ), "Number of previously rewarded ports differ in some mice!"
        rewarded_licks = np.zeros((n_mice, licks.shape[1], n_rewarded[0]))
        previously_rewarded_licks = np.zeros(
            (n_mice, licks.shape[1], n_previously_rewarded[0])
        )

        # If the current sessions' rewards don't match the day before's,
        # the remaining ports is total ports minus currently rewarded
        # minus previously rewarded.
        if not same_rewards:
            remainder = 8 - n_rewarded[0] - n_previously_rewarded[0]
        # Otherwise, since they overlap, it's total ports minus
        # current ports (which are the same as the day before's).
        else:
            remainder = 8 - n_rewarded[0]
        remainder = 0 if remainder < 0 else remainder
        other_licks = np.zeros((n_mice, licks.shape[1], remainder))

        # For each mouse, find the rewarded and non-rewarded ports.
        # Place them into the appropriate array.
        for mouse, (
            rewarded_ports_in_this_mouse,
            previously_rewarded_ports_in_this_mouse,
        ) in enumerate(zip(rewarded_ports, previously_rewarded)):
            rewarded_licks[mouse] = licks[mouse, :, rewarded_ports_in_this_mouse].T

            previously_rewarded_licks[mouse] = licks[
                mouse, :, previously_rewarded_ports_in_this_mouse
            ].T

            if remainder > 0:
                other_licks[mouse] = licks[
                    mouse,
                    :,
                    (
                        ~rewarded_ports_in_this_mouse
                        & ~previously_rewarded_ports_in_this_mouse
                    ),
                ].T

        # Plot these lick data.
        licks_to_plot = [rewarded_licks]
        colors = ["cornflowerblue"]
        # If the previous day's rewards are different, add them to the list.
        if not same_rewards:
            licks_to_plot.append(previously_rewarded_licks)
            colors.append("lightcoral")
        # If there are any more ports to plot, add them to the list.
        if remainder > 0:
            licks_to_plot.append(other_licks)
            colors.append("gray")

        # Plot rewarded and non-rewarded ports in different colors.
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.3, 4.8))
        for licks_in_this_category, color in zip(licks_to_plot, colors):
            # Take the mean across mice and trials.
            mean_across_mice = np.nanmean(licks_in_this_category, axis=0)
            mean_across_trials = np.nanmean(mean_across_mice, axis=1)

            # To calculate the standard error, treat all the ports
            # in the same category (rewarded or non-rewarded) as
            # different samples. The n will actually be number of
            # mice multiplied by number of ports in that category.
            stacked_ports = (
                licks_in_this_category[:, :, port]
                for port in range(licks_in_this_category.shape[2])
            )
            reshaped = np.vstack(stacked_ports)
            standard_error = sem(reshaped, axis=0)

            # Plot.
            trials = range(mean_across_trials.shape[0])  # x-axis
            errorfill(trials, mean_across_trials, standard_error, color=color, ax=ax)
            ax.set_xlabel("Trial #")
            ax.set_ylabel("Licks")

    def get_previous_rewards(self, session_type):
        """
        Gets the rewarded ports from the session previous to the one
        specified.

        :param session_type:
        :return:
        """
        # If it's the first session, don't get the previous session.
        previous_session_number = self.meta["session_types"].index(session_type) - 1
        assert previous_session_number > -1, KeyError(
            "No other session before this one!"
        )

        previous_session = self.meta["session_types"][previous_session_number]
        previously_rewarded = self.data["rewarded_ports"][previous_session]

        return previously_rewarded


if __name__ == "__main__":
    mice = [
        # "Betelgeuse_Scope25",
        # "Alcor_Scope20",
        "Castor_Scope05",
        # "Draco_Scope02",
        # "Encedalus_Scope14",
         "Fornax",
         "Hydra",
         "Io",
        # "M1",
        # "M2",
        # "M3",
        # "M4",
    ]
    # B = BatchBehaviorAnalyses(mice)
    # B.plot_learning_trials_per_mouse()
    # B.plot_all_session_licks()
    # B.plot_all_sdts(1)
    # B.compare_d_prime(8, 'CircleTrackReversal1', 'CircleTrackReversal2')

    B = BatchFullAnalyses(mice)
    B.correlate_stability_to_reversal()
    #B.spiral_scrollplot_assemblies('Castor_Scope05', 'CircleTrackReversal1')
    lapsed_assemblies, spiking = B.plot_lapsed_assemblies('Castor_Scope05', ('CircleTrackGoals2','CircleTrackReversal1'))

    pass