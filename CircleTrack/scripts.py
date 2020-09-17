from CircleTrack.BehaviorFunctions import *
import matplotlib.pyplot as plt
from CaImaging.util import sem, errorfill
from grid_strategy.strategies import RectangularStrategy
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches

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
    data = BehaviorSession(folder)
    data.port_approaches(acceleration=accleration, window=window)


def PlotBlockedApproaches(folder, acceleration=True, blocks=4):
    data = BehaviorSession(folder)
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
        self.mice = mice
        self.n_mice = len(mice)

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
        self.licks, self.rewarded_ports, self.n_drinks, self.p_drinks \
            = self.resort_data()
        self.learning_trials = self.get_learning_trials()

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
        for sessions in self.all_sessions.values():
            for session in sessions.values():
                session.SDT(n_trial_blocks=n_trial_blocks,
                            plot=False)

        # Resort data into session types, listed by mouse.
        sdt_matrix = dict()
        sdt_categories = ['hits', 'misses', 'FAs', 'CRs', 'd_prime']
        for session_type in self.session_types:
            sdt_matrix[session_type] = {key: nan_array((self.n_mice,
                                                        n_trial_blocks))
                                        for key in sdt_categories}

            for m, mouse in enumerate(self.mice):
                mouse_data = self.all_sessions[mouse]

                try:
                    for key in sdt_categories:
                        sdt_matrix[session_type][key][m] = \
                            mouse_data[session_type].sdt[key]
                except KeyError:
                    print(f'{session_type} not found for '
                          f'mouse {mouse}! Skipping...')

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
        hit_patch = mpatches.Patch(color='forestgreen',
                                   label='Hits')
        cr_patch = mpatches.Patch(color='steelblue',
                                  label='Correct rejections')
        dprime_patch = mpatches.Patch(color='k', label="d'")
        # Handle the case where you want the entire session's hit/
        # correct rejection rate/d'. Plots two subplots, one of
        # hit/correct rejection rate, another for d' across all sessions.
        if n_trial_blocks == 1:
            # Preallocate.
            fig, axs = plt.subplots(2, 1, figsize=(7,9.5))
            hits = nan_array((self.n_mice, len(self.session_types)))
            CRs = nan_array((self.n_mice, len(self.session_types)))
            d_primes = nan_array((self.n_mice, len(self.session_types)))

            # Acquire data and sort by session.
            for s, (session_type, label) \
                    in enumerate(zip(self.session_types,
                                     self.session_labels)):
                self.verify_sdt(1, session_type, 'hits')
                self.verify_sdt(1, session_type, 'CRs')
                self.verify_sdt(1, session_type, 'd_prime')

                for m, mouse in enumerate(self.mice):
                    try:
                        session_data = \
                            self.all_sessions[mouse][session_type]

                        hits[m, s] = session_data.sdt['hits'][0]
                        CRs[m, s] = session_data.sdt['CRs'][0]
                        d_primes[m, s] = session_data.sdt['d_prime'][0]
                    except KeyError:
                        print(f'{session_type} not found for '
                              f'mouse {mouse}!')

            # Plot the values.
            axs[0].plot(self.session_labels, hits.T, 'o-',
                        color='forestgreen', label='Hits')
            axs[0].set_ylabel('%')
            axs[0].plot(CRs.T, 'o-', color='steelblue',
                        label='Correct rejections')
            axs[1].plot(self.session_labels, d_primes.T, 'o-',
                        color='black')
            axs[1].set_ylabel("d'")
            for ax in axs:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout()

            axs[0].legend(handles=[hit_patch, cr_patch])

        # Otherwise, split the sessions into trial blocks and
        # plot d' etc per block.
        else:
            # Preallocate the figure axes.
            fig, axs = plt.subplots(4,2, sharey='all', figsize=(7,9.5))
            d_prime_axs = []
            for ax, session_type, label in zip(axs.flatten(),
                                               self.session_types,
                                               self.session_labels):
                self.plot_sdt('hits', n_trial_blocks, session_type, ax)

                # Ignore shaping sessions.
                if 'Shaping' not in session_type:
                    self.plot_sdt('CRs', n_trial_blocks, session_type, ax)

                    # Use a different axis for d'.
                    d_prime_ax = ax.twinx()
                    d_prime_axs.append(d_prime_ax)
                    self.plot_sdt('d_prime', n_trial_blocks,
                                  session_type, d_prime_ax)
                ax.set_title(label)

            # Link the d' axes.
            for d_prime_ax in d_prime_axs[1:]:
                d_prime_axs[0].get_shared_y_axes().join(d_prime_axs[0],
                                                        d_prime_ax)
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
        if not hasattr(self, 'sdt'):
            self.sdt = self.signal_detection_analysis(n_trial_blocks)
        elif self.sdt[session_type][category].shape[1] != n_trial_blocks:
            self.sdt = self.signal_detection_analysis(n_trial_blocks)


    def compare_d_prime(self, n_trial_blocks,
                        session1_type, session2_type):
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
        self.verify_sdt(n_trial_blocks, session1_type, 'd_prime')
        self.verify_sdt(n_trial_blocks, session2_type, 'd_prime')

        # Get the data from those sesssions and mice.
        session1 = self.sdt[session1_type]['d_prime']
        session2 = self.sdt[session2_type]['d_prime']

        # Get the best way to arrange subplots. Plot all d's.
        # Also record p-values of signed-rank tests and
        # get their multiple comparisons corrected values.
        grid = RectangularStrategy.get_grid_arrangement(n_trial_blocks)
        fig, axs = plt.subplots(max(grid), len(grid),
                                sharex='all', sharey='all',
                                figsize=(6.5,7.5))
        flattened_axs = axs.flatten()
        p_vals = []
        for ax, block in zip(flattened_axs, range(session1.shape[1])):
            ax.scatter(session1[:, block], session2[:, block])
            p = wilcoxon(session1[:, block], session2[:, block]).pvalue
            p_vals.append(p)
        corrected_ps = multipletests(p_vals, method='fdr_bh')[1]

        # Get size of y=x line.
        ax = flattened_axs[0]
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # Get x and y axis labels.
        labels = [self.session_labels[self.session_types.index(session_type)]
                  for session_type in [session1_type, session2_type]]

        # Labels. 
        for ax, p, cp in zip(flattened_axs,
                             p_vals,
                             corrected_ps):
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.set_xlabel(f"{labels[0]} d'")
            ax.set_ylabel(f"{labels[1]} d'")
            ax.set_title(f'p = {str(np.round(p, 2))}, '
                         f'{str(np.round(cp, 2))}')
        fig.tight_layout()

        return session1, session2


    def plot_sdt(self, category, n_trial_blocks,
                  session_type, ax=None):
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
        colors = {'hits': 'forestgreen',
                  'misses': 'darkred',
                  'FAs': 'goldenrod',
                  'CRs': 'steelblue',
                  'd_prime': 'black'}

        # For spacing apart data points.
        jitter = {'hits': -.1,
                  'misses': -.1,
                  'FAs': .1,
                  'CRs': .1,
                  'd_prime': 0}

        if ax is None:
            fig, ax = plt.subplots()

        # Get the x and y axis values.
        plot_me = self.sdt[session_type][category].T
        trial_number = np.asarray(list(range(n_trial_blocks)), dtype=float)
        trial_number += jitter[category]

        ax.plot(trial_number, plot_me, 'o-', color=colors[category],
                alpha=0.6)
        ax.set_xlabel('Trial blocks')

        # Only put % on the left side of the plot.
        if 'd_prime' not in category:
            ax.set_ylabel('%')


    def get_learning_trials(self):
        learning_trials = {'start': nan_array((self.n_mice,
                                               len(self.session_types))),
                           'inflection': nan_array((self.n_mice,
                                                    len(self.session_types)))}

        for s, session_type in enumerate(self.session_types):
            for m, mouse in enumerate(self.mice):
                mouse_data = self.all_sessions[mouse]
                try:
                    learning_trials['start'][m, s] = \
                        mouse_data[session_type].learning['start']
                    learning_trials['inflection'][m, s] = \
                        mouse_data[session_type].learning['inflection']
                except KeyError:
                    pass

        return learning_trials


    def plot_learning_trials_across_sessions(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.session_labels,
                self.learning_trials['inflection'].T, 'yo-')
        ax.set_ylabel('Learning inflection trial #')


    def plot_learning_trials_per_mouse(self):
        for mouse in self.mice:
            mouse_data = self.all_sessions[mouse]
            fig, axs = plt.subplots(3,2, sharex='all', figsize=(6.4, 6))

            sessions = [session for session in self.session_types
                        if 'Shaping' not in session]
            for ax, session in zip(axs.flatten(),
                                   sessions):
                try:
                    mouse_data[session].plot_learning_curve(ax=ax)
                except KeyError:
                    pass

            fig.tight_layout(pad=0.5)

            start_patch = mpatches.Patch(color='g',
                                         label='Start of learning')
            inflection_patch = mpatches.Patch(color='y',
                                              label='Inflection point')
            axs.flatten()[-1].legend(handles=[start_patch,
                                              inflection_patch])
        pass


    def count_trials(self):
        """
        Count the number of trials for each mouse and each session type.

        :return:
        """
        trial_counts = nan_array((self.n_mice,
                                  len(self.session_types)))
        for s, session_type in enumerate(self.session_types):
            for m, mouse in enumerate(self.mice):
                mouse_data = self.all_sessions[mouse]
                try:
                    trial_counts[m,s] = int(mouse_data[session_type].ntrials)
                except KeyError:
                    trial_counts[m,s] = np.nan

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
        ax.plot(self.session_labels, self.trial_counts.T, 'o-')
        ax.set_xlabel('Session type')
        ax.set_ylabel('Number of trials')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
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
        for max_trials, session_type in zip(self.max_trials,
                                            self.session_types):
            lick_matrix[session_type] = nan_array((self.n_mice,
                                                   max_trials,
                                                   8))
            rewarded_matrix[session_type] = np.zeros((self.n_mice,
                                                      8), dtype=bool)
            drink_matrix[session_type] = nan_array((self.n_mice,
                                                    max_trials))
            p_drink_matrix[session_type] = nan_array((self.n_mice,
                                                      max_trials))
            learning_start[session_type] = nan_array((self.n_mice,))
            learning_inflection[session_type] = nan_array((self.n_mice,))

            # Get data and sort by session type. .
            for m, mouse in enumerate(self.mice):
                mouse_data = self.all_sessions[mouse]
                try:
                    session_licks = mouse_data[session_type].all_licks
                    mat_size = session_licks.shape
                    lick_matrix[session_type][m, :mat_size[0], :mat_size[1]] = \
                        session_licks

                    # Also get which ports were rewarded.
                    rewarded = mouse_data[session_type].rewarded_ports
                    rewarded_matrix[session_type][m] = \
                        rewarded

                    # Also get number of drinks for each trial.
                    session_drinks = mouse_data[session_type].n_drinks
                    drink_matrix[session_type][m, :session_drinks.shape[0]] = \
                        session_drinks

                    # If the session is called 'Shaping', mark
                    # all ports as rewarded. Some ports get marked
                    # as non-rewarded sometimes because they were never
                    # visited due mouse shyness (⁄ ⁄•⁄ω⁄•⁄ ⁄)
                    if 'Shaping' in session_type and not all(rewarded):
                        print('Non-rewarded ports found during a '
                              'shaping session. Setting all ports '
                              'to rewarded')
                        rewarded_matrix[session_type][m] = \
                            np.ones_like(rewarded_matrix, dtype=bool)

                    # And percentage of water deliveries out of all rewarded ports.
                    n_rewarded_ports = np.sum(rewarded_matrix[session_type][m])
                    p_drink_matrix[session_type][m, :session_drinks.shape[0]] = \
                        session_drinks / n_rewarded_ports
                except KeyError:
                    print(f'{session_type} not found for mouse {mouse}! Skipping...')

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
        fig, axs = plt.subplots(4,2,sharey='all', sharex='all',
                                figsize=(6.6, 9.4))
        for ax, session_type, label in zip(axs.flatten(),
                                           self.session_types,
                                           self.session_labels):
            self.plot_rewarded_licks(session_type, ax)
            ax.set_title(label)

        # Build the legend.
        rewarded_patch = mpatches.Patch(color='cornflowerblue',
                                        label='Rewarded')
        prev_rewarded_patch = mpatches.Patch(color='lightcoral',
                                             label='Rewarded last session')
        not_rewarded_patch = mpatches.Patch(color='gray',
                                            label='Not recently rewarded')
        axs.flatten()[-1].legend(handles=[rewarded_patch,
                                          prev_rewarded_patch,
                                          not_rewarded_patch])

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
        licks = self.licks[session_type]
        rewarded_ports = self.rewarded_ports[session_type]

        # Find the previous session to figure out what was previously rewarded.
        try:
            previously_rewarded = self.get_previous_rewards(session_type)
        except AssertionError:
            previously_rewarded = np.ones_like(rewarded_ports, dtype=bool)

        # If the rewards from the last session match this session,
        # we're going to treat this a little differently.
        same_rewards = True \
            if np.all(previously_rewarded == rewarded_ports) \
            else False
        if np.any(previously_rewarded == rewarded_ports) and not same_rewards:
            print(f'Warning! At least one reward port for {session_type} overlaps '
                  'with the previous day. Are you sure this is correct?')

        # Some mice might not have the specified session.
        # Exclude those mice.
        mice_to_include = [session_type in mouse
                           for mouse in self.all_sessions.values()]
        n_mice = np.sum(mice_to_include)
        licks = licks[mice_to_include]
        rewarded_ports = rewarded_ports[mice_to_include]
        previously_rewarded = previously_rewarded[mice_to_include]

        # Find the number of rewarded ports to allocate
        # two arrays -- one each for rewarded, previously rewarded,
        # and non-rewarded licks.
        n_rewarded = np.unique(np.sum(rewarded_ports, axis=1))
        n_previously_rewarded = np.unique(np.sum(previously_rewarded, axis=1))
        assert len(n_rewarded)==1, \
            'Number of rewarded ports differ in some mice!'
        assert len(n_previously_rewarded)==1, \
            'Number of previously rewarded ports differ in some mice!'
        rewarded_licks = np.zeros((n_mice,
                                   licks.shape[1],
                                   n_rewarded[0]))
        previously_rewarded_licks = np.zeros((n_mice,
                                              licks.shape[1],
                                              n_previously_rewarded[0]))

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
        other_licks = np.zeros((n_mice,
                                licks.shape[1],
                                remainder))

        # For each mouse, find the rewarded and non-rewarded ports.
        # Place them into the appropriate array.
        for mouse, (rewarded_ports_in_this_mouse,
                    previously_rewarded_ports_in_this_mouse) \
                in enumerate(zip(rewarded_ports, previously_rewarded)):
            rewarded_licks[mouse] = \
                licks[mouse, :, rewarded_ports_in_this_mouse].T

            previously_rewarded_licks[mouse] = \
                licks[mouse, :, previously_rewarded_ports_in_this_mouse].T

            if remainder > 0:
                other_licks[mouse] = \
                    licks[mouse, :, (~rewarded_ports_in_this_mouse
                                     & ~previously_rewarded_ports_in_this_mouse)].T

        # Plot these lick data.
        licks_to_plot = [rewarded_licks]
        colors = ['cornflowerblue']
        # If the previous day's rewards are different, add them to the list.
        if not same_rewards:
            licks_to_plot.append(previously_rewarded_licks)
            colors.append('lightcoral')
        # If there are any more ports to plot, add them to the list.
        if remainder > 0:
            licks_to_plot.append(other_licks)
            colors.append('gray')

        # Plot rewarded and non-rewarded ports in different colors.
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.3, 4.8))
        for licks_in_this_category, color \
                in zip(licks_to_plot, colors):
            # Take the mean across mice and trials.
            mean_across_mice = np.nanmean(licks_in_this_category, axis=0)
            mean_across_trials = np.nanmean(mean_across_mice, axis=1)

            # To calculate the standard error, treat all the ports
            # in the same category (rewarded or non-rewarded) as
            # different samples. The n will actually be number of
            # mice multiplied by number of ports in that category.
            stacked_ports = (licks_in_this_category[:,:,port]
                             for port in range(licks_in_this_category.shape[2]))
            reshaped = np.vstack(stacked_ports)
            standard_error = sem(reshaped, axis=0)

            # Plot.
            trials = range(mean_across_trials.shape[0])     # x-axis
            errorfill(trials, mean_across_trials,
                      standard_error, color=color, ax=ax)
            ax.set_xlabel('Trials')
            ax.set_ylabel('Licks')


    def get_previous_rewards(self, session_type):
        """
        Gets the rewarded ports from the session previous to the one
        specified.

        :param session_type:
        :return:
        """
        # If it's the first session, don't get the previous session.
        previous_session_number = self.session_types.index(session_type) - 1
        assert previous_session_number > -1, \
            KeyError('No other session before this one!')

        previous_session = self.session_types[previous_session_number]
        previously_rewarded = self.rewarded_ports[previous_session]

        return previously_rewarded




if __name__ == '__main__':
    B = BatchBehaviorAnalyses(['Betelgeuse_Scope25',
                               'Alcor_Scope20',
                               'Castor_Scope05',
                               'M1',
                               'M2',
                               'M3',
                               'M4'])
    B.plot_learning_trials_per_mouse()
    B.plot_all_session_licks()
    B.plot_all_sdts(1)
    B.compare_d_prime(8, 'CircleTrackReversal1', 'CircleTrackReversal2')