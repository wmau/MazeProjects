from CircleTrack.BehaviorFunctions import BehaviorSession, spiral_plot
import matplotlib.pyplot as plt
import numpy as np
from CaImaging.util import sync_data
from util import Session_Metadata
from CircleTrack.BehaviorFunctions import linearize_trajectory
from CaImaging.util import ScrollPlot
from CircleTrack.plotting import plot_spiral
from CaImaging.PlaceFields import PlaceFields

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
        self.BehaviorSession = BehaviorSession(self.folder)

        # Get paths
        self.minian_path = self.BehaviorSession.paths["minian"]
        timestamp_paths = self.BehaviorSession.paths["timestamps"]

        # Combine behavioral and calcium imaging data.
        self.data = dict()
        self.data["behavior"], self.data["imaging"], _ = sync_data(
            self.BehaviorSession.behavior_df, self.minian_path, timestamp_paths
        )

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
        t = np.asarray(self.data["behavior"].frame)

        # Linearize position.
        lin_position = np.asarray(linearize_trajectory(self.data["behavior"])[0])

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


    def linearized_activity(self, bin_size=10):
        PFs = PlaceFields(np.asarray(self.data["behavior"]["lin_position"]),
                          np.zeros_like(self.data["behavior"]["lin_position"]),
                          self.data["imaging"]["S"],
                          bin_size_cm=0.02,
                          one_dim=True)
        pass


if __name__ == "__main__":
    folder = r"Z:\Will\Drift\Data\Castor_Scope05\09_09_2020_CircleTrackGoals2\16_46_11"
    S = CalciumSession(folder)
    S.linearized_activity()
    pass
