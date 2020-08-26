from CircleTrack.BehaviorFunctions import *
import matplotlib.pyplot as plt

from CircleTrack.BehaviorFunctions import MultiSession

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False
plt.rcParams.update({'font.size': 12})

def PlotApproaches(folder, accleration=True, window=(-15,15)):
    """
    Plot approach speeds to each water port.

    :param folder:
    :param accleration:
    :return:
    """
    data = Session(folder)
    data.port_approaches(acceleration=accleration, window=window)


def PlotBlockedApproaches(folder, acceleration=True, blocks=4):
    data = Session(folder)
    data.blocked_port_approaches()



class BatchBehaviorAnalyses:
    def __init__(self, mice, project_folder=r'Z:\Will\Drift\Data'):
        """
        This class definition will contain behavior analyses spanning
        the entire dataset (or at least the specified mice).

        :param mice:
        :param project_folder:
        """
        # Compile data for all animals.
        self.all_sessions = MultiAnimal(mice, project_folder,
                                        behavior='CircleTrack')
        self.n_mice = len(mice)

        # Get the number of licks for every single session.
        for animal, sessions in self.all_sessions.items():
            for session_type, session in sessions.items():
                session.get_licks(plot=False)

        # Define session types here. Watch out for typos.
        # Order matters. Plots will be in the order presented here.
        self.session_types = ['CircleTrackShaping1',
                              'CircleTrackShaping2',
                              'CircleTrackGoals1',
                              'CircleTrackGoals2',
                              'CircleTrackReversal1',
                              'CircleTrackReversal2',
                              'CircleTrackRecall']

        # Same as session types, just without 'CircleTrack'.
        self.session_labels = [session_type.replace('CircleTrack', '')
                               for session_type in self.session_types]

        # Gather the number of trials for each session type and
        # arrange it in a (mouse, session) array.
        self.trial_counts, self.max_trials = self.count_trials()
        self.gather_licks()

        pass


    def count_trials(self):
        """
        Count the number of trials for each mouse and each session type.

        :return:
        """
        trial_counts = nan_array((self.n_mice,
                                  len(self.session_types)))
        for s, session_type in enumerate(self.session_types):
            for a, animal in enumerate(self.all_sessions.values()):
                try:
                    trial_counts[a,s] = animal[session_type].ntrials
                except KeyError:
                    trial_counts[a,s] = np.nan

        # Get the highest number of trials across all mice for a particular
        # session type.
        max_trials = np.nanmax(trial_counts, axis=0)

        return trial_counts, max_trials


    def plot_trials(self):
        """
        Plot number of trials for each mouse over the experiment.

        :return:
        """
        fig, ax = plt.subplots()
        ax.plot(self.session_labels, self.trial_counts.T, 'o-')
        ax.set_xlabel('Session type')
        ax.set_ylabel('Number of trials')
        plt.tight_layout()
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


    def gather_licks(self):
        """
        For each mouse, session type, and trial, get the number of licks.
        :return:
        """

        lick_matrix = nan_array((self.n_mice,
                                 self.max_trials, # Different for each session type!
                                 8))
        for m, mouse in enumerate(self.all_sessions.values()):
            for s, session in enumerate(mouse.values()):
                pass
                #lick_matrix[m, :session.all_licks.shape[0], ]




    #def arrange_licking(self):





if __name__ == '__main__':
    B = BatchBehaviorAnalyses(['Betelgeuse_Scope25', 'Alcor_Scope20', 'M1', 'M2', 'M3', 'M4'])