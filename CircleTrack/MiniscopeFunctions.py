from CircleTrack.BehaviorFunctions import BehaviorSession
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import sync_data, nan_array
from CaImaging.Miniscope import get_transient_timestamps
from util import Session_Metadata
from CircleTrack.BehaviorFunctions import linearize_trajectory
from CaImaging.util import ScrollPlot
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
from CircleTrack.utils import sync, get_equivalent_local_path
from CircleTrack.Assemblies import (
    write_assembly_triggered_movie,
    plot_assembly,
    find_members,
)


hv.extension("bokeh")
from bokeh.plotting import show
from itertools import product
from scipy.stats import spearmanr, zscore
from matplotlib import colors


class CalciumSession:
    def __init__(
        self,
        session_folder,
        spatial_bin_size_radians=0.05,
        S_std_thresh=1,
        velocity_threshold=7,
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
            "velocity_threshold": velocity_threshold,
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

    def plot_spiral_spikes(self, first_neuron=0):
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
        spikes = self.imaging["S_binary"]
        t = np.asarray(self.behavior.data["df"]["frame"])

        # Linearize position.
        lin_position = np.asarray(linearize_trajectory(self.behavior.data["df"])[0])

        # Do the show_plot.
        cell_number_labels = [f"Cell #{n}" for n, _ in enumerate(spikes)]
        self.spiral_spatial_plot = ScrollPlot(
            plot_spiral,
            current_position=first_neuron,
            t=t,
            lin_position=lin_position,
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

        tuning_curves = self.spatial.data["placefields"][neurons]

        cell_number_labels = [f"Cell #{n}" for n in neurons]
        self.raster_plot = ScrollPlot(
            plot_raster,
            nrows=2,
            rasters=rasters,
            tuning_curves=tuning_curves,
            binary=binary,
            titles=cell_number_labels,
            figsize=(3, 6.5),
        )

        return self.raster_plot

    def spiral_scrollplot_assemblies(self, threshold=2):
        assemblies = self.assemblies
        behavior_df = self.behavior.data["df"]

        z_activation = zscore(assemblies["activations"], axis=1)
        above_threshold = z_activation > threshold

        titles = [
            f"Assembly #{n}" for n in range(assemblies["significance"].nassemblies)
        ]
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
            color = 'g' if rewarded else 'k'
            ax.axvline(x=port, color=color)

        return ax

    def spatial_activity_by_trial(self):
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
        bin_edges = spatial_bin(
            lin_position,
            filler,
            bin_size_cm=self.meta["spatial_bin_size"],
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
                positions_this_trial, filler, bins=bin_edges, one_dim=True
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
        corr_matrix: (trial, trial) np.array
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

        fpath = os.path.join(self.meta["folder"], f"Assembly #{assembly_number}.avi")
        write_assembly_triggered_movie(
            assembly_activations,
            behavior_frame_numbers,
            movie_fname,
            fpath=fpath,
            threshold=threshold,
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

        fig = plt.figure()
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
        activation_ax.set_title(f'Ensemble # {assembly_number}')

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
                pattern,
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


def nan_corrupted_frames(miniscope_folder, C, S, frames):
    bad_frames_folder = os.path.join(miniscope_folder, "bad_frames")
    if os.path.exists(bad_frames_folder):
        bad_frames = [
            int(os.path.splitext(fname)[0]) for fname in os.listdir(bad_frames_folder)
        ]
        n_frames = len(bad_frames)
        print(f"{n_frames} bad frames found. Correcting...")

        match = np.asarray([x in bad_frames for x in frames])
        C[:, match] = np.nan
        S[:, match] = np.nan
    else:
        pass

    return C, S


if __name__ == "__main__":
    folder = (r"Z:\Will\RemoteReversal\Data\Fornax\2021_02_25_Reversal\10_04_05",)

    S = CalciumSession(folder)
    pvals = S.spatial["placefield_class"].data["spatial_info_pvals"]
    S.scrollplot_rasters(neurons=np.where(np.asarray(pvals) < 0.01)[0], binary=True)
    # S.spatial_activity_by_trial(0.1)
    # S.correlate_spatial_PVs_by_trial()
    pass
