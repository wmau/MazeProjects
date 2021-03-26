from CircleTrack.BehaviorFunctions import BehaviorSession
from CircleTrack.MiniscopeFunctions import CalciumSession
import matplotlib.pyplot as plt
from CaImaging.util import (
    sem,
    nan_array,
    bin_transients,
    make_bins,
    ScrollPlot,
    contiguous_regions,
)
from CaImaging.plotting import errorfill
from scipy.stats import spearmanr
from CircleTrack.SessionCollation import MultiAnimal
from CaImaging.CellReg import rearrange_neurons, trim_map, scrollplot_footprints
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression
from scipy.stats import circmean
import numpy as np
from scipy.stats import mode, zscore
import os
from CircleTrack.plotting import plot_daily_rasters
from CaImaging.Assemblies import preprocess_multiple_sessions, lapsed_activation
from CircleTrack.Assemblies import plot_assembly
from itertools import product
import xarray as xr
import pymannkendall as mk

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 12})


class RemoteReversal:
    def __init__(self, mice, behavior_only=True):
        if behavior_only:
            SessionFunction = BehaviorSession
        else:
            SessionFunction = CalciumSession

        self.mice = mice
        self.meta = {"session_types":
                        ["Goals1",
                         "Goals2",
                         "Goals3",
                         "Goals4",
                         "RecentReversal"]
                     }
        self.data = MultiAnimal(mice, project_name='RemoteReversal',
                                SessionFunction=SessionFunction)

        pass

class BatchFullAnalyses:
    def __init__(self, mice):
        # Collect data from all mice and sessions.
        self.data = MultiAnimal(
            mice, project_name="Drift", SessionFunction=CalciumSession
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

        return overlaps

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
        session_pairs=(('CircleTrackGoals1', 'CircleTrackGoals2'),
                       ('CircleTrackGoals2', 'CircleTrackReversal1'))
    ):
        r = {key: [] for key in session_pairs}
        for mouse in self.meta["mice"]:
            for key, pair in zip(session_pairs, session_pairs):
                r[key].append(
                    self.correlate_fields(mouse, pair, show_histogram=False)["r"]
                )

        fig, ax = plt.subplots()
        for i, data in enumerate(r.values()):
            x = np.linspace(2*i, 2*i+1, len(self.meta['mice']))
            plot_me = [np.asarray(mouse_data)[~np.isnan(mouse_data)] for mouse_data in data]
            ax.boxplot(plot_me, positions=x)

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

    def find_assembly_trends(self, mouse, session_type, x='time', z_threshold=2.58, x_bin_size=60):
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
        z_activations = zscore(assemblies['activations'], axis=1)

        # Binarize activations so that there are no strings of 1s spanning multiple frames.
        binarized_activations = np.zeros_like(z_activations)
        for i, assembly in enumerate(z_activations):
            on_frames = contiguous_regions(assembly > z_threshold)[:,0]
            binarized_activations[i, on_frames] = 1

        # If binning by time, sum activations every few seconds or minutes.
        if x == 'time':
            binned_activations = []

            for assembly in binarized_activations:
                binned_activations.append(bin_transients(assembly[np.newaxis], x_bin_size, fps=15)[0])

            binned_activations = np.vstack(binned_activations)

        # If binning by trials, sum activations every n trials.
        elif x == 'trial':
            trial_bins = np.arange(0, session.behavior.data['ntrials'], x_bin_size)
            df = session.behavior.data['df']

            binned_activations = np.zeros((binarized_activations.shape[0], len(trial_bins)))
            for i, assembly in enumerate(binarized_activations):
                for j, (lower, upper) in enumerate(zip(trial_bins, trial_bins[1:])):
                    in_trial = (df['trials'] >= lower) & (df['trials'] < upper)
                    binned_activations[i, j] = np.sum(assembly[in_trial])

        else:
            raise ValueError('Invalid value for x.')

        # Group assemblies into either increasing, decreasing, or no trend in occurrence rate.
        assembly_trends = {'no trend': [],
                           'decreasing': [],
                           'increasing': [],
                           }
        for i, assembly in enumerate(binned_activations):
            mk_test = mk.original_test(assembly)
            assembly_trends[mk_test.trend].append(i)

        return assembly_trends, binned_activations

    def plot_assembly_trends(self, show_plot=True):
        session_types = self.meta['session_types']
        session_labels = self.meta['session_labels']
        assembly_trend_arr = np.zeros(
            (3,
             len(self.meta['mice']),
             len(session_types),
             ),
            dtype=object
                                         )
        assembly_counts_arr = np.zeros_like(assembly_trend_arr)

        trend_strs = ['no trend', 'decreasing', 'increasing']
        for i, mouse in enumerate(self.meta['mice']):
            for j, session_type in enumerate(session_types):
                assembly_trends = self.find_assembly_trends(mouse, session_type)

                for h, trend in enumerate(trend_strs):
                    assembly_trend_arr[h, i, j] = assembly_trends[trend]
                    assembly_counts_arr[h, i, j] = len(assembly_trends[trend])

        assembly_trends = xr.DataArray(assembly_trend_arr,
                                       dims=('trend', 'mouse', 'session'),
                                       coords={'trend': trend_strs,
                                               'mouse': self.meta['mice'],
                                               'session': session_types,
                                               })

        assembly_counts = xr.DataArray(assembly_counts_arr,
                                       dims=('trend', 'mouse', 'session'),
                                       coords={'trend': trend_strs,
                                               'mouse': self.meta['mice'],
                                               'session': session_types,
                                               })

        if show_plot:
            fig, ax = plt.subplots()
            p_decreasing = assembly_counts.sel(trend='decreasing') / assembly_counts.sum(dim='trend')

            ax.plot(session_labels, p_decreasing.T)
            ax.set_ylabel('Proportion ensembles with decreasing activity over session')
            plt.setp(ax.get_xticklabels(), rotation=45)

        return assembly_trends, assembly_counts


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
    # B = BatchFullAnalyses(mice)
    # B.correlate_stability_to_reversal()
    # #B.spiral_scrollplot_assemblies('Castor_Scope05', 'CircleTrackReversal1')
    # lapsed_assemblies, spiking = B.plot_lapsed_assemblies('Castor_Scope05', ('CircleTrackGoals2','CircleTrackReversal1'))

    RR = RemoteReversal(['Fornax',
                         # 'Gemini_aged',
                         # #'Io',
                         # 'Janus',
                         # 'Lyra',
                         # 'Miranda',
                         # 'Naiad',
                         # 'Oberon_aged',
                         #'Puck_aged'
                         ]
                        )
    RR.data['Fornax']['Goals3'].sdt_trials()
    pass