import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt
from LinearTrack.BehaviorFunctions import BehaviorSession
from util import Session_Metadata, find_timestamp_file
from LinearTrack.plotting import plot_raster, plot_directional_raster
from CircleTrack.utils import sync, get_equivalent_local_path, find_reward_spatial_bins
from CaImaging.Miniscope import get_transient_timestamps, nan_corrupted_frames
from CaImaging.PlaceFields import PlaceFields
from CaImaging.Behavior import spatial_bin
from CaImaging.util import nan_array, ScrollPlot


class CalciumSession:
    def __init__(
            self,
            session_folder,
            spatial_bin_size=1.905, # ~1.905 cm for 50 bins, use None
            nbins=None,
            S_std_thresh=1,
            velocity_threshold=4,
            place_cell_alpha=0.001,
            place_cell_transient_threshold='n_trials',
            overwrite_synced_data=False,
            overwrite_placefields=False,
            overwrite_placefield_trials=False,
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
            "spatial_bin_size": spatial_bin_size,
            "nbins": nbins,
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
            self.behavior = BehaviorSession(self.meta["folder"],
                                            pix_per_cm=6.3)

            # Get paths
            self.meta["paths"] = self.behavior.meta["paths"]
            if not self.meta["paths"]["minian"]:
                meta = Session_Metadata(session_folder, overwrite=True)
                self.meta["paths"]["minian"] = meta.meta_dict["minian"]
            timestamp_paths = self.meta["paths"]["timestamps"]

            # Combine behavioral and calcium imaging data.
            self.behavior.data["df"], self.imaging = sync(
                self.meta["paths"]["minian"],
                self.behavior.data["df"], timestamp_paths,
                convert_to_np=False
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
                self.spatial.meta["bin_size"] == spatial_bin_size,
                self.spatial.meta["nbins"] == nbins,
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
                np.zeros_like(self.behavior.data["df"]["x"]),
                self.imaging["S"],
                bin_size=self.meta['spatial_bin_size'],
                nbins=self.meta['nbins'],
                circular=False,
                linearized=True,
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
        lin_position = np.asarray(behavior_df["x"])
        running = self.spatial.data["running"]
        filler = np.zeros_like(lin_position)
        if nbins is None:
            nbins = self.meta['nbins']
        bin_edges = spatial_bin(
            lin_position,
            filler,
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
            positions_this_trial = behavior_df.loc[time_bins, "x"]
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
        # Total number of transients across all laps for each spatial bin.
        n_transients = np.sum(np.sum(self.spatial.data['rasters'], axis=1), axis=1)
        place_cells = np.where(np.logical_and(self.spatial.data['spatial_info_pvals'] < alpha,
                                              n_transients> transient_threshold))[0]

        return place_cells

    def scrollplot_rasters(self, neurons=None, binary=True):
        if neurons is None:
            neurons = range(self.imaging["n_neurons"])

        rasters = self.spatial.data["rasters"][neurons]
        if binary:
            rasters = rasters > 0

        tuning_curves = self.spatial.data["placefields_normalized"][neurons]

        cell_number_labels = [f"Cell #{n}" for n in neurons]
        self.raster_plot = ScrollPlot(
            plot_raster,
            nrows=2,
            rasters=rasters,
            tuning_curves=tuning_curves,
            binary=binary,
            titles=cell_number_labels,
            figsize=(5, 8),
        )

        return self.raster_plot

    def scrollplot_directional_rasters(self, neurons=None):
        if neurons is None:
            neurons = range(self.imaging["n_neurons"])

        rasters = self.spatial.data["rasters"][neurons]
        directions = self.behavior.data["df"]["direction"]
        trials = self.behavior.data["df"]["trials"]

        split_rasters = {}
        tuning_curves = {}
        for direction in ['left','right']:
            xward_trials = np.unique(trials[directions==direction])
            split_rasters[direction] = nan_array((len(neurons), len(xward_trials), rasters.shape[2]))
            tuning_curves[direction] = nan_array((len(neurons), rasters.shape[2]))

            for i, raster in enumerate(rasters):
                split_rasters[direction][i] = raster[xward_trials]
                tuning_curves[direction][i] = np.nanmean(raster[xward_trials], axis=0)

        cell_number_labels = [f"Cell #{n}" for n in neurons]

        self.raster_plot = ScrollPlot(
            plot_directional_raster,
            subplot_kw={"sharey": "row"},
            nrows=2,
            ncols=2,
            rasters=split_rasters,
            tuning_curves=tuning_curves,
            titles=cell_number_labels,
            figsize=(10, 8),
        )

        pass