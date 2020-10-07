from CircleTrack.BehaviorFunctions import BehaviorSession, spiral_plot
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import sync_data, nan_array
from CaImaging.Miniscope import threshold_S
from util import Session_Metadata
from CircleTrack.BehaviorFunctions import linearize_trajectory
from CaImaging.util import ScrollPlot
from CircleTrack.plotting import plot_spiral
from CaImaging.PlaceFields import PlaceFields
from CaImaging.Behavior import spatial_bin
import holoviews as hv
hv.extension('bokeh')
from bokeh.plotting import show


class CalciumSession:
    def __init__(self, session_folder):
        """
        Single session analyses and plots for miniscope data.

        :parameter
        session_folder: str
            Session folder.
        """
        # Get the behavioral data.
        self.folder = session_folder
        self.data = {"behavior": BehaviorSession(self.folder)}

        # Get paths
        self.minian_path = self.data["behavior"].paths["minian"]
        timestamp_paths = self.data["behavior"].paths["timestamps"]

        # Combine behavioral and calcium imaging data.
        self.data["behavior"].behavior_df, self.data["imaging"], _ = sync_data(
            self.data["behavior"].behavior_df, self.minian_path, timestamp_paths
        )

        # Get number of neurons.
        self.n_neurons = self.data['imaging']['C'].shape[0]

        self.spatial = dict()
        self.spatial['fields'], self.spatial['occupancy'] = self.spatial_activity_by_trial()

    def plot_spiral_spikes(self, first_neuron=0, S_thresh=1):
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
        spikes = [
            activity > (np.mean(activity) + S_thresh * np.std(activity))
            for activity in self.data["imaging"]["S"]
        ]
        t = np.asarray(self.data["behavior"].behavior_df['frame'])

        # Linearize position.
        lin_position = np.asarray(linearize_trajectory(self.data["behavior"].behavior_df)[0])

        # Do the plot.
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


    def linearized_activity(self, bin_size=0.02):
        PFs = PlaceFields(np.asarray(self.data["behavior"]["lin_position"]),
                          np.zeros_like(self.data["behavior"]["lin_position"]),
                          self.data["imaging"]["S"],
                          bin_size_cm=bin_size,
                          one_dim=True)
        pass


    def spatial_activity_by_trial(self, bin_size_radians=0.02):
        """
        Plot activity trial by trial, binned in linearized space.

        :param bin_size_radians:
        :return:
        """
        # Get linearized position. We'll also need a dummy variable
        # for the binning function because it usually takes 2d.
        behavior_df = self.data["behavior"].behavior_df
        lin_position = np.asarray(behavior_df["lin_position"])
        filler = np.zeros_like(lin_position)
        bin_edges = spatial_bin(lin_position, filler,
                                bin_size_cm=bin_size_radians,
                                one_dim=True)[1]

        # Threshold S matrix here.
        S = threshold_S(self.data["imaging"]["S"], std_thresh=1)

        # For each trial, spatial bin position weighted by S.
        occ_map_by_trial = []
        fields = nan_array((self.n_neurons,
                            self.data["behavior"].ntrials,
                            len(bin_edges)-1))
        for trial_number in range(self.data["behavior"].ntrials):
            time_bins = behavior_df['trials'] == trial_number
            positions_this_trial = behavior_df.loc[time_bins, 'lin_position']
            filler = np.zeros_like(positions_this_trial)

            # Get occupancy this trial.
            occupancy = spatial_bin(positions_this_trial,
                                    filler,
                                    bins=bin_edges,
                                    one_dim=True)[0]

            # Weight position by activity of each neuron.
            for n, neuron in enumerate(S):
                spiking = neuron[time_bins]
                fields[n, trial_number, :] = spatial_bin(positions_this_trial,
                                                         filler,
                                                         bins=bin_edges,
                                                         one_dim=True,
                                                         weights=spiking)[0]


            occ_map_by_trial.append(occupancy)
        occ_map_by_trial = np.vstack(occ_map_by_trial)

        return fields, occ_map_by_trial


    def viz_spatial_trial_activity(self, bin_size_radians=0.02, neurons=range(10),
                                   preserve_neuron_idx=True):
        fields = self.spatial_activity_by_trial(bin_size_radians=bin_size_radians)[0]

        if preserve_neuron_idx:
            viz_fields = {n: hv.Image(fields[n] > 0).opts(cmap='gray')
                          for n in neurons}
        else:
            viz_fields = {i: hv.Image(fields[n] > 0).opts(cmap='gray')
                          for i, n in enumerate(neurons)}

        return viz_fields


if __name__ == "__main__":
    folder = r"Z:\Will\Drift\Data\Castor_Scope05\09_09_2020_CircleTrackGoals2\16_46_11"
    S = CalciumSession(folder)
    S.spatial_activity_by_trial(0.1)
    pass
