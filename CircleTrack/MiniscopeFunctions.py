from CircleTrack.BehaviorFunctions import BehaviorSession
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import nan_array, ScrollPlot, \
    sync_cameras_v4
from CaImaging.Miniscope import get_transient_timestamps, \
    nan_corrupted_frames
from util import Session_Metadata, find_timestamp_file
from CircleTrack.BehaviorFunctions import linearize_trajectory, make_tracking_video
from CircleTrack.plotting import plot_spiral, plot_raster, spiral_plot
from CaImaging.PlaceFields import PlaceFields
from CaImaging.Behavior import spatial_bin
import holoviews as hv
import os
import pickle as pkl
from matplotlib import gridspec
from CaImaging.Assemblies import (
    find_assemblies,
    preprocess_multiple_sessions,
    membership_sort,
    plot_assemblies,
)
from CircleTrack.utils import sync, get_equivalent_local_path, find_reward_spatial_bins
from CircleTrack.Assemblies import (
    write_assembly_triggered_movie,
    plot_assembly,
    find_members,
)

hv.extension("bokeh")
from itertools import product
from scipy.stats import spearmanr, zscore


class CalciumSession:
    def __init__(
        self,
        session_folder,
        spatial_bin_size_radians=0.05,
        S_std_thresh=1,
        velocity_threshold=7,
        place_cell_alpha=0.001,
        place_cell_transient_threshold='n_trials',
        overwrite_synced_data=False,
        overwrite_placefields=False,
        overwrite_placefield_trials=False,
        overwrite_assemblies=False,
        local=True,
    ):
        """
        Single session analyses and plots for miniscope data.

        :parameter
        session_folder: str
            Session minian_folder.
        """
        # Get the metadata.
        self.meta = {
            "folder": session_folder,
            "spatial_bin_size": spatial_bin_size_radians,
            "S_std_thresh": S_std_thresh,
            "threshold": velocity_threshold,
            "local": local,
        }

        #############################################
        # Get the synced behavior and calcium imaging data.
        fpath = self.get_pkl_path("SyncedData.pkl")

        try:
            if overwrite_synced_data:
                print(f"Overwriting {fpath}.")
                raise Exception
            with open(fpath, "rb") as file:
                (self.behavior, self.imaging) = pkl.load(file)

            self.meta["paths"] = self.behavior.meta["paths"]
        except:
            self.behavior = BehaviorSession(self.meta["folder"])

            # Get paths
            self.meta["paths"] = self.behavior.meta["paths"]
            if not self.meta["paths"]["minian"]:
                meta = Session_Metadata(session_folder, overwrite=True)
                self.meta["paths"]["minian"] = meta.meta_dict["minian"]
            timestamp_paths = self.meta["paths"]["timestamps"]

            # Combine behavioral and calcium imaging data.
            self.behavior.data["df"], self.imaging = sync(
                self.meta["paths"]["minian"], self.behavior.data["df"], timestamp_paths
            )

            self.imaging["C"], self.imaging["S"] = self.nan_bad_frames()

            # Redo trial counting to account in case some frames got cut
            # from behavior (e.g. because a miniscope video got truncated).
            self.behavior.data["ntrials"] = max(self.behavior.data["df"]["trials"] + 1)

            with open(fpath, "wb") as file:
                pkl.dump((self.behavior, self.imaging), file)

        (
            self.imaging["spike_times"],
            self.imaging["spike_mags"],
            self.imaging["S_binary"],
        ) = get_transient_timestamps(self.imaging["S"], thresh_type="eps")

        # Get number of neurons.
        self.imaging["n_neurons"] = self.imaging["C"].shape[0]

        #############################################
        # Get place fields.
        fpath = self.get_pkl_path("Placefields.pkl")
        rerun_placefield_trials = False
        try:
            if overwrite_placefields:
                print(f"Overwriting {fpath}.")
                raise Exception

            with open(fpath, "rb") as file:
                self.spatial = pkl.load(file)

            parameters_match = [
                self.spatial.meta["bin_size"] == spatial_bin_size_radians,
                self.spatial.meta["velocity_threshold"] == velocity_threshold,
            ]

            if not all(parameters_match):
                print("A placefield parameter does not match saved data, rerunning.")
                rerun_placefield_trials = True
                raise Exception

        except:
            self.spatial = PlaceFields(
                np.asarray(self.behavior.data["df"]["t"]),
                np.asarray(self.behavior.data["df"]["x"]),
                np.asarray(self.behavior.data["df"]["y"]),
                self.imaging["S"],
                bin_size=self.meta["spatial_bin_size"],
                circular=True,
                fps=self.behavior.meta["fps"],
                shuffle_test=True,
                velocity_threshold=velocity_threshold,
            )

            with open(fpath, "wb") as file:
                pkl.dump(self.spatial, file)

            #############################################
        # Get spatial activity by trial.
        fpath = self.get_pkl_path("PlacefieldTrials.pkl")
        try:
            if overwrite_placefield_trials or rerun_placefield_trials:
                print(f"Overwriting {fpath}")
                raise Exception

            with open(fpath, "rb") as file:
                (
                    self.spatial.data["rasters"],
                    self.spatial.data["trial_occupancy"],
                ) = pkl.load(file)
        except:
            (
                self.spatial.data["rasters"],
                self.spatial.data["trial_occupancy"],
            ) = self.spatial_activity_by_trial()

            with open(fpath, "wb") as file:
                pkl.dump(
                    (
                        self.spatial.data["rasters"],
                        self.spatial.data["trial_occupancy"],
                    ),
                    file,
                )

            #############################################
        # Get assemblies.
        fpath = self.get_pkl_path("Assemblies.pkl")
        try:
            if overwrite_assemblies:
                print(f"Overwriting {fpath}")
                raise Exception
            with open(fpath, "rb") as file:
                self.assemblies = pkl.load(file)

        except:
            processed_for_assembly_detection = preprocess_multiple_sessions(
                [self.imaging["S"]], smooth_factor=5, use_bool=True
            )
            data = processed_for_assembly_detection["processed"][0]
            self.assemblies = find_assemblies(
                data, nullhyp="circ", plot=False, n_shuffles=500
            )

            with open(fpath, "wb") as file:
                pkl.dump(self.assemblies, file)

        if place_cell_transient_threshold == 'n_trials':
            place_cell_transient_threshold = self.behavior.data['ntrials']
        self.spatial.data['place_cells'] = self.get_place_cells(alpha=place_cell_alpha,
                                                                transient_threshold=place_cell_transient_threshold)
        self.spatial.meta['place_cell_pval'] = place_cell_alpha
        self.spatial.meta['place_cell_transient_threshold'] = place_cell_transient_threshold

    def get_pkl_path(self, fname):
        if self.meta["local"]:
            local_path = get_equivalent_local_path(self.meta["folder"])
            fpath = os.path.join(local_path, fname)
        else:
            fpath = os.path.join(self.meta["folder"], fname)

        return fpath

    def nan_bad_frames(self):
        miniscope_folder = self.meta["paths"]["minian"]
        C = self.imaging["C"]
        S = self.imaging["S"]
        frames = self.imaging["frames"]

        corrected_C, corrected_S = nan_corrupted_frames(miniscope_folder, C, S, frames)

        return corrected_C, corrected_S

    def spiralplot_spikes(self, neurons=None):
        """
        Plot where on the maze a neuron spikes, starting with
        first_neuron. Scroll through the rest.

        :parameter
        first_neuron: int
            Neuron to start plotting.

        S_thresh: float
            Number of standard deviations above the mean to consider
            a spike.
        """
        # Get spiking activity and time vector.
        if neurons is None:
            neurons = range(self.imaging['S_binary'])
        spikes = np.vstack([spikes_i for spikes_i in self.imaging["S_binary"][neurons]])
        t = np.asarray(self.behavior.data["df"]["frame"])

        # Linearize position.
        lin_position = np.asarray(linearize_trajectory(self.behavior.data["df"])[0])

        # Do the show_plot.
        cell_number_labels = [f"Cell #{n}" for n in neurons]
        self.spiral_spatial_plot = ScrollPlot(
            plot_spiral,
            t=t,
            lin_position=lin_position,
            lin_ports=self.behavior.data['lin_ports'],
            rewarded=self.behavior.data['rewarded_ports'],
            markers=spikes,
            marker_legend="Spikes",
            subplot_kw={"projection": "polar"},
            titles=cell_number_labels,
        )

    def scrollplot_rasters(self, neurons=None, binary=True):
        if neurons is None:
            neurons = range(self.imaging["n_neurons"])

        rasters = self.spatial.data["rasters"][neurons]
        if binary:
            rasters = rasters > 0
            cmap = 'binary'
        else:
            cmap = 'gray'

        tuning_curves = self.spatial.data["placefields_normalized"][neurons]

        behavior_data = self.behavior.data

        port_bins = find_reward_spatial_bins(behavior_data['df']['lin_position'],
                                               np.asarray(behavior_data['lin_ports']),
                                               spatial_bin_size_radians=self.spatial.meta['bin_size'])[0]

        cell_number_labels = [f"Cell #{n}" for n in neurons]
        self.raster_plot = ScrollPlot(
            plot_raster,
            nrows=2,
            rasters=rasters,
            tuning_curves=tuning_curves,
            port_bins=port_bins,
            rewarded=behavior_data['rewarded_ports'],
            cmap=cmap,
            interpolation='none',
            titles=cell_number_labels,
            figsize=(5, 8),
        )

        return self.raster_plot

    def spiral_scrollplot_assemblies(self, threshold=2, order=None):
        assemblies = self.assemblies
        behavior_df = self.behavior.data["df"]

        if order is None:
            order = range(assemblies["significance"].nassemblies)

        z_activation = zscore(assemblies["activations"], axis=1)
        above_threshold = z_activation > threshold
        above_threshold = above_threshold[order]

        titles = [f"Assembly #{n}" for n in order]
        ScrollPlot(
            plot_spiral,
            t=behavior_df["t"],
            lin_position=behavior_df["lin_position"],
            markers=above_threshold,
            marker_legend="Assembly activation",
            subplot_kw={"projection": "polar"},
            lin_ports=self.behavior.data["lin_ports"],
            rewarded=self.behavior.data["rewarded_ports"],
            titles=titles,
        )

    def spiralplot_assembly(self, assembly_number, threshold=2, ax=None):
        behavior_data = self.behavior.data
        above_threshold = zscore(self.assemblies['activations'][assembly_number]) > threshold

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar")
        ax = spiral_plot(t=behavior_data['df']['t'],
                         lin_position=behavior_data['df']['lin_position'],
                         markers=above_threshold,
                         ax=ax,
                         marker_legend='Ensemble activation')
        ax.set_title(f"Ensemble {assembly_number}")
        for rewarded, port in zip(behavior_data['rewarded_ports'], behavior_data['lin_ports']):
            color = 'g' if rewarded else 'gray'
            ax.axvline(x=port, color=color)

        return ax

    def get_ensemble_field_COM(self, ensemble_number):
        COM = np.average(self.behavior.data['df']['lin_position'],
                         weights=self.assemblies['activations'][ensemble_number])

        return COM

    def spatial_activity_by_trial(self, nbins=None):
        """
        Plot activity trial by trial, binned in linearized space.

        :parameter
        ---
        bin_size_radians: float
            Spatial bin size in radians.

        """
        # Get linearized position. We'll also need a dummy variable
        # for the binning function because it usually takes 2d.
        behavior_df = self.behavior.data["df"]
        lin_position = np.asarray(behavior_df["lin_position"])
        running = self.spatial.data["running"]
        filler = np.zeros_like(lin_position)
        if nbins is None:
            nbins = self.spatial.meta['nbins']
            bin_size = None
        else:
            bin_size = self.meta["spatial_bin_size"]

        bin_edges = spatial_bin(
            lin_position,
            filler,
            bin_size_cm=bin_size,
            nbins=nbins,
            one_dim=True,
        )[1]

        # Threshold S matrix here.
        S = self.imaging["S_binary"]

        # For each trial, spatial bin position weighted by S.
        occ_map_by_trial = []
        fields = nan_array(
            (
                self.imaging["n_neurons"],
                self.behavior.data["ntrials"],
                len(bin_edges) - 1,
            )
        )
        for trial_number in range(self.behavior.data["ntrials"]):
            time_bins = behavior_df["trials"] == trial_number
            positions_this_trial = behavior_df.loc[time_bins, "lin_position"]
            filler = np.zeros_like(positions_this_trial)
            running_this_trial = running[time_bins]

            # Get occupancy this trial.
            occupancy = spatial_bin(
                positions_this_trial, filler, bins=bin_edges, one_dim=True,
                weights=running_this_trial.astype(int)
            )[0]

            # Weight position by activity of each neuron.
            for n, neuron in enumerate(S):
                spiking = (neuron[time_bins] * running_this_trial).astype(int)
                fields[n, trial_number, :] = spatial_bin(
                    positions_this_trial,
                    filler,
                    bins=bin_edges,
                    one_dim=True,
                    weights=spiking,
                )[0]

            occ_map_by_trial.append(occupancy)
        occ_map_by_trial = np.vstack(occ_map_by_trial)

        return fields, occ_map_by_trial

    def get_place_cells(self, alpha=0.001, transient_threshold=40):
        n_transients = np.sum(np.sum(self.spatial.data['rasters'], axis=1), axis=1)
        place_cells = np.where(np.logical_and(self.spatial.data['spatial_info_pvals'] < alpha,
                               n_transients> transient_threshold))[0]

        return place_cells

    def placefield_reliability(self, neuron, field_threshold=0.15, even_split=True, split=4,
                               show_plot=True):
        """
        Compute the trial-by-trial in-field consistency of a neuron (whether it fired in field).

        :parameters
        ---
        neuron: int
            Neuron number.

        field_threshold: float
            Percentage of field peak to be considered part of the field.

        even_split: bool
            Flag to split trials evenly into n parts (defined by the argument split). If False,
            instead split is the number of trials in each split.

        split: int
            If even_split, the number of parts to divide the trial into.
            If not even_split, the number of trials in each split.

        """
        raster = self.spatial.data['rasters'][neuron]
        placefield = self.spatial.data['placefields_normalized'][neuron]

        field_bins = np.where(placefield > max(placefield) * field_threshold)[0]
        fired_in_field = np.any(raster[:, field_bins], axis=1)

        if not even_split:
            split = np.arange(split, len(fired_in_field), split)

        split_fired_in_field = np.array_split(fired_in_field, split)

        # Handles cases where you don't actually want to split the reliability across trials.
        if split == 1 and even_split:
            reliability = np.sum(split_fired_in_field[0])/len(split_fired_in_field[0])
        else:
            reliability = [np.sum(fired_in_field) / len(fired_in_field) for fired_in_field in split_fired_in_field]

        if show_plot and split != 1:
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(raster > 0, aspect='auto')
            axs[0].set_xlabel('Position')
            axs[0].set_ylabel('Trials')

            axs[1].plot(reliability, range(len(reliability)))
            axs[1].set_xlim([0, 1])
            axs[1].invert_yaxis()
            axs[1].set_ylabel(f'Trial block #, {len(split_fired_in_field[0])} trials per block')
            axs[1].set_xlabel('Proportion of trials with '
                              '\n in-field calcium transient')
            fig.tight_layout()

        return reliability

    def port_reliability(self, neuron, even_split=False, splits=6, show_plot=True):
        spatial_bins = np.linspace(0, 2*np.pi, 8)
        S_binary = self.imaging['S_binary'][neuron]
        df = self.behavior.data['df']
        ntrials = self.behavior.data['ntrials']
        in_bin = np.digitize(df['lin_position'],
                             spatial_bins, right=False)

        reliability_matrix = nan_array((ntrials, 8))
        for trial in range(ntrials):
            on_trial = df['trials'] == trial
            for bin in range(8):
                inds = on_trial & (in_bin==bin)
                reliability_matrix[trial, bin] = np.sum(S_binary[inds])

        return reliability_matrix

    def viz_spatial_trial_activity(self, neurons=range(10), preserve_neuron_idx=True):
        """
        Visualize single cell activity binned in spatial and separated
        by trials.

        :parameters
        ---
        bin_size_radians: float
            Spatial bin size in radians.

        neurons: array-like of ints
            List of neuron indices.

        preserve_neuron_idx: boolean
            In the output dict, viz_fields, this flag either keeps the
            neuron index or reassigns all neurons to new dict keys
            starting from 0. The latter option is useful for neurons
            registered across days and serves as a global index for
            Holomap to access.
        """
        fields = self.spatial_activity_by_trial()[0]

        if preserve_neuron_idx:
            viz_fields = {n: hv.Image(fields[n] > 0).opts(cmap="gray") for n in neurons}
        else:
            viz_fields = {
                i: hv.Image(fields[n] > 0).opts(cmap="gray")
                for i, n in enumerate(neurons)
            }

        return viz_fields

    def correlate_spatial_PVs_by_trial(self, show_plot=True):
        """
        Correlate trial pairs of spatial ratemaps.

        :return
        corr_matrices: (trial, trial) np.array
            Correlation matrix.
        """
        # Change the axes of the fields from (cell, trial, spatial bin)
        # to (trial, cell, spatial bin).
        # fields = np.asarray([field / self.spatial['occupancy'] for field in self.spatial['fields']])
        fields = self.spatial.data["rasters"]
        fields_by_trial = np.rollaxis(fields, 1)

        # Compute spearman correlation coefficient.
        corr_matrix = np.zeros((fields_by_trial.shape[0], fields_by_trial.shape[0]))
        for i, (a, b) in enumerate(product(fields_by_trial, repeat=2)):
            idx = np.unravel_index(i, corr_matrix.shape)
            corr_matrix[idx] = spearmanr(a.flatten(), b.flatten(), nan_policy="omit")[0]

        # Fill diagonal with 0s.
        np.fill_diagonal(corr_matrix, 0)
        # offset = colors.DivergingNorm(vcenter=0)

        if show_plot:
            fig, ax = plt.subplots()
            ax.imshow(corr_matrix, cmap="bwr", vmin=-1, vmax=1)

        return corr_matrix

    def write_assembly_activation_movie(self, assembly_number, threshold=2):
        assembly_activations = self.assemblies["activations"][assembly_number]
        behavior_frame_numbers = self.behavior.data["df"]["frame"].to_numpy()
        movie_fname = self.meta["paths"]["BehaviorVideo"]
        trials = self.behavior.data["df"]["trials"].to_numpy()

        fpath = os.path.join(self.meta["folder"], f"Assembly #{assembly_number}.avi")
        write_assembly_triggered_movie(
            assembly_activations,
            behavior_frame_numbers,
            movie_fname,
            fpath=fpath,
            threshold=threshold,
            trials=trials,
        )

    def plot_assembly(self, assembly_number, neurons=None, get_members=True, filter_method='sd', thresh=2):
        pattern = self.assemblies["patterns"][assembly_number]
        n_neurons = len(pattern)
        activation = self.assemblies["activations"][assembly_number]
        spike_times_ = self.imaging["spike_times"]

        # This option lets you only plot the ensemble members.
        if get_members:
            members, corrected_pattern = find_members(
                pattern, filter_method=filter_method, thresh=thresh
            )[1:]
            sort_by_contribution = False
        else:
            members = np.arange(n_neurons)
            sort_by_contribution = True

        if neurons is None:
            if get_members:
                included_neurons = members
            else:
                included_neurons = np.arange(n_neurons)
        else:
            if not get_members:
                included_neurons = neurons
            else:
                in_neurons = np.isin(members, neurons)
                excluded_members = members[~in_neurons]

                # If there are member neurons that were not in the neurons list, let me know.
                print("Excluded ensemble members: " + str(excluded_members))
                included_neurons = members[in_neurons]

        fig = plt.figure(figsize=(10.5, 7.5))
        spec = gridspec.GridSpec(nrows=1, ncols=6, figure=fig)
        assembly_ax = fig.add_subplot(spec[:5])
        pattern_ax = fig.add_subplot(spec[-1])
        fig.subplots_adjust(wspace=1)
        spike_times = [spike_times_[neuron] for neuron in included_neurons]
        activation_ax, spikes_ax = plot_assembly(
            pattern,
            activation,
            spike_times,
            sort_by_contribution=sort_by_contribution,
            order=None,
            ax=assembly_ax,
        )
        activation_ax.set_title(f'Ensemble # {assembly_number}', fontsize=22)

        if get_members:
            n_members = len(members)
            markerlines_members, stemlines_members = pattern_ax.stem(
                range(n_members),
                corrected_pattern[-n_members:][::-1],
                "r",
                orientation="horizontal",
                basefmt=" ",
                markerfmt="ro",
            )[:2]
            markerlines, stemlines = pattern_ax.stem(
                np.arange(n_members, len(corrected_pattern)),
                corrected_pattern[:-n_members][::-1],
                "b",
                orientation="horizontal",
                basefmt=" ",
                markerfmt="bo",
            )[:2]
            plt.setp(stemlines_members, 'linewidth', 1)
            plt.setp(markerlines_members, 'markersize', 1)
            plt.setp(stemlines, 'linewidth', 1)
            plt.setp(markerlines, 'markersize', 1)
        else:
            markerlines, stemlines = pattern_ax.stem(
                range(len(pattern)),
                np.sort(pattern),
                "b",
                orientation="horizontal",
                basefmt=" ",
                markerfmt="bo",
            )[:2]
            plt.setp(stemlines, 'linewidth', 1)
            plt.setp(markerlines, 'markersize', 1)

        pattern_ax.invert_yaxis()
        pattern_ax.axis("off")

        return activation_ax, spikes_ax

    def plot_all_assemblies(self):
        sorted_spiking, sorted_colors = membership_sort(
            self.assemblies["patterns"], self.imaging["spike_times"]
        )
        plot_assemblies(
            self.assemblies["activations"], sorted_spiking, colors=sorted_colors
        )

    def sync_behavior_movie(self, miniscope_vids=(40, 43)):
        """
        Concatenates frames in the behavior video to be synchronous with
        the specified miniscope videos.

        :parameter
        ---
        miniscope_vids: (2,) tuple
            The miniscope video numbers that the behavior movie will be
            synchronized with. Doesn't include the second value.

        """
        timestamp_fpath = self.meta['paths']['timestamps']
        miniscope_file = find_timestamp_file(timestamp_fpath, "Miniscope")
        behavior_file = find_timestamp_file(timestamp_fpath, "BehavCam")
        sync_map, DAQ_data = sync_cameras_v4(miniscope_file, behavior_file)

        frame_window = [vid * 1000 for vid in miniscope_vids]
        frames = np.asarray(sync_map['fmCam1'][np.arange(frame_window[0],
                                                         frame_window[1])])

        make_tracking_video(self.meta['folder'],
                            output_fname=f'{miniscope_vids[0]}_{miniscope_vids[1]}.avi',
                            frames=frames,
                            fps=60)


if __name__ == "__main__":
    folder = r"Z:\Will\RemoteReversal\Data\Fornax\2021_02_24_Goals4\09_06_18"
    S = CalciumSession(folder)
    #pvals = S.spatial["placefield_class"].data["spatial_info_pvals"]
    #S.scrollplot_rasters(neurons=np.where(np.asarray(pvals) < 0.01)[0], binary=True)
    # S.spatial_activity_by_trial(0.1)
    # S.correlate_spatial_PVs_by_trial()
    S.plot_all_assemblies()
    pass
