import numpy as np
from CircleTrack.SessionCollation import MultiAnimal
from CircleTrack.BehaviorFunctions import BehaviorSession
import matplotlib.pyplot as plt
import xarray as xr
from CaImaging.util import nan_array, sem
import pandas as pd
from CaImaging.plotting import errorfill, beautify_ax, jitter_x
import matplotlib.patches as mpatches

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
    "PSAMReversal": ["Goals1", "Goals2", "Goals3", "Goals4", "Reversal"],
}

# Exclude PSAM4, PSAM5, and PSAM15 because they never learned
# (poor Goals4 performance).
PSEM_mice = ['PSAM_2',
             'PSAM_3',
             'PSAM_5',
             'PSAM_6',
             'PSAM_7',
             'PSAM_8',
             'PSAM_10',
             'PSAM_13',
             'PSAM_17',
             'PSAM_19',
             'PSAM_20',
             'PSAM_24',
             'PSAM_25',
             'PSAM_26',
             'PSAM_27']

aged_mice = [
    "Gemini",
    "Oberon",
    "Puck",
    "Umbriel",
    "Virgo",
    "Ymir",
    "Atlas",
    "PSAM_1",
    "PSAM_2",
    "PSAM_3",
]

ages = ["young", "aged"]
PSAM_groups = ["vehicle", "PSEM"]
age_colors = ["cornflowerblue", "r"]
PSAM_colors = ['silver', 'coral']

class PSAM:
    def __init__(self, mice):
        self.data = MultiAnimal(mice, project_name='PSAMReversal',
                                SessionFunction=BehaviorSession)



        self.meta = {
            "session_types": session_types['PSAMReversal'],
            "mice": mice,
        }

        self.meta["session_labels"] = [
            session_type.replace("Goals", "Training")
            for session_type in self.meta["session_types"]
        ]
        self.meta["grouped_mice"] = {
            "PSEM": [mouse for mouse in self.meta["mice"] if mouse in PSEM_mice],
            "vehicle": [mouse for mouse in self.meta["mice"] if mouse not in PSEM_mice],
        }

    def plot_all_behavior(
            self,
            window=6,
            strides=2,
            ax=None,
            performance_metric="d_prime",
            show_plot=True,
            trial_limit=None,
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
                session = self.data[mouse][session_type]
                session.sdt_trials(
                    rolling_window=window,
                    trial_interval=strides,
                    plot=False,
                    trial_limit=trial_limit,
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
            inj: (len(self.meta["grouped_mice"][inj]), sum(longest_sessions))
            for inj in PSAM_groups
        }
        metrics = {key: nan_array(dims[key]) for key in PSAM_groups}

        for inj in PSAM_groups:
            for row, mouse in enumerate(self.meta["grouped_mice"][inj]):
                for border, session in zip(borders, self.meta["session_types"]):
                    metric_this_session = behavioral_performance.sel(
                        metric=performance_metric, mouse=mouse, session=session
                    ).values.tolist()
                    length = len(metric_this_session)
                    metrics[inj][row, border : border + length] = metric_this_session

        if show_plot:
            ylabels = {
                "d_prime": "d'",
                "CRs": "Correct rejection rate",
                "hits": "Hit rate",
            }
            if ax is None:
                fig, ax = plt.subplots(figsize=(7.4, 5.7))
            else:
                fig = ax.figure

            if window is not None:
                for inj, c in zip(PSAM_groups, PSAM_colors):
                    ax.plot(metrics[inj].T, color=c, alpha=0.3)
                    errorfill(
                        range(metrics[inj].shape[1]),
                        np.nanmean(metrics[inj], axis=0),
                        sem(metrics[inj], axis=0),
                        ax=ax,
                        color=c,
                        label=inj,
                    )

                for session in borders[1:]:
                    ax.axvline(x=session, color="k")
                ax.set_xticks(borders)
                ax.set_xticklabels(np.insert(longest_sessions, 0, 0))
                ax.set_xlabel("Trial blocks")
                _ = beautify_ax(ax)
            else:
                for inj, c in zip(PSAM_groups, PSAM_colors):
                    ax.plot(metrics[inj].T, color=c, alpha=0.3)
                    ax.errorbar(
                        self.meta['session_labels'],
                        np.nanmean(metrics[inj], axis=0),
                        sem(metrics[inj], axis=0),
                        color=c,
                        label=inj,
                        capsize=5,
                        linewidth=3,
                    )
            ax.set_ylabel(ylabels[performance_metric])
            fig.legend()

        if window is None:
            mice_ = np.hstack([np.repeat(self.meta['grouped_mice'][inj],
                                         len(self.meta['session_types']))
                               for inj in PSAM_groups])
            groups_ = np.hstack([np.repeat(inj, metrics[inj].size) for inj in PSAM_groups])
            session_types_ = np.hstack([np.tile(self.meta['session_types'],
                                                len(self.meta['grouped_mice'][inj]))
                                        for inj in PSAM_groups])
            metric_ = np.hstack([metrics[inj].flatten() for inj in PSAM_groups])

            df = pd.DataFrame(
                {'metric': metric_,
                 'session_types': session_types_,
                 'mice': mice_,
                 'group': groups_,
                 }
            )
        else:
            df = None

        return behavioral_performance, metrics, df

    def plot_best_performance(
            self,
            session_type,
            ax=None,
            window=None,
            performance_metric="d_prime",
            show_plot=True,
            downsample_trials=False,
    ):
        if downsample_trials:
            trial_limit = min(
                [
                    self.data[mouse][session_type].data["ntrials"]
                    for mouse in self.meta["mice"]
                ]
            )
        else:
            trial_limit = None

        behavioral_performance = self.plot_all_behavior(
            show_plot=False,
            window=window,
            performance_metric=performance_metric,
            trial_limit=trial_limit,
        )[0]

        best_performance = dict()
        for inj in PSAM_groups:
            best_performance[inj] = []
            for mouse in self.meta["grouped_mice"][inj]:
                best_performance[inj].append(
                    np.nanmax(
                        behavioral_performance.sel(
                            mouse=mouse, metric=performance_metric, session=session_type
                        ).values.tolist()
                    )
                )

        if show_plot:
            label_axes = True
            ylabels = {
                'CRs': 'Correct rejection rate',
                'hits': 'Hit rate',
                'd_prime': "d'"
            }
            if ax is None:
                fig, ax = plt.subplots(figsize=(3, 4.75))
            else:
                label_axes = False
            box = ax.boxplot(
                [best_performance[inj] for inj in PSAM_groups],
                labels=PSAM_groups,
                patch_artist=True,
                widths=0.75,
                showfliers=False,
                zorder=0,
            )

            [ax.scatter(
                jitter_x(np.ones_like(best_performance[inj])*(i+1), 0.1),
                best_performance[inj],
                color=color,
                edgecolor='k',
                zorder=1,
            )
                for i, (inj, color) in enumerate(zip(PSAM_groups, PSAM_colors))]
            for patch, med, color in zip(
                    box["boxes"], box["medians"], PSAM_colors
            ):
                patch.set_facecolor(color)
                med.set(color="k")

            if label_axes:
                ax.set_xticks([1, 2])
                ax.set_xticklabels(PSAM_groups)
                ax.set_ylabel(ylabels[performance_metric])
                #ax = beautify_ax(ax)
                plt.tight_layout()

        return best_performance

    def plot_best_performance_all_sessions(
            self,
            window=None,
            performance_metric="CRs",
            downsample_trials=False,
            sessions=None,
    ):
        if sessions is None:
            sessions = self.meta['session_types']

        session_labels = [self.meta['session_labels'][self.meta['session_types'].index(session)]
                          for session in sessions]
        fig, axs = plt.subplots(1, len(sessions), sharey=True)
        fig.subplots_adjust(wspace=0)
        ylabels = {
            "d_prime": "d'",
            "CRs": "Correct rejection rate",
            "hits": "Hit rate",
        }
        performance = dict()
        for ax, session, title in zip(
                axs, sessions, session_labels
        ):
            performance[session] = self.plot_best_performance(
                session_type=session,
                ax=ax,
                window=window,
                performance_metric=performance_metric,
                downsample_trials=downsample_trials,
            )
            ax.set_xticks([])
            ax.set_title(title)
        axs[0].set_ylabel(ylabels[performance_metric])

        patches = [
            mpatches.Patch(facecolor=c, label=label, edgecolor="k")
            for c, label in zip(PSAM_colors, PSAM_groups)
        ]
        fig.legend(handles=patches, loc="lower right")

        return performance

    def plot_behavior_grouped(self,
                              window=None,
                              performance_metric='d_prime',
                              sessions=None):
        if sessions is None:
            sessions = self.meta['session_types']

        session_labels = [self.meta['session_labels'][self.meta['session_types'].index(session)]
                          for session in sessions]

        performance_all = nan_array((len(self.meta['mice']),
                                     len(sessions)))

        for i, session in enumerate(sessions):
            performance = self.plot_best_performance(
                session_type=session,
                window=window,
                performance_metric=performance_metric,
                show_plot=False,
            )

            performance['vehicle'].extend(performance['PSEM'])
            performance_all[:,i] = performance['vehicle']

        ylabel = {
            'CRs': 'Correct rejection rate',
            'hits': 'Hit rate',
            'd_prime': "d'"
        }
        fig, ax = plt.subplots(figsize=(4,5))
        ax.plot(performance_all.T, color='gray', alpha=0.5)
        errorfill(session_labels, np.mean(performance_all, axis=0),
                  sem(performance_all, axis=0), ax=ax, color='k')
        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        ax.set_ylabel(ylabel[performance_metric])
        fig.tight_layout()


    def plot_perseverative_licking(self, show_plot=True, binarize=True):
        goals4 = "Goals4"
        reversal = "Reversal"

        perseverative_errors = dict()
        unforgiveable_errors = dict()
        for inj in PSAM_groups:
            perseverative_errors[inj] = []
            unforgiveable_errors[inj] = []

            for mouse in self.meta["grouped_mice"][inj]:
                behavior_data = self.data[mouse][reversal].data

                if binarize:
                    licks = behavior_data["all_licks"] > 0
                else:
                    licks = behavior_data["all_licks"]

                previous_reward_ports = self.data[mouse][goals4].data[
                    "rewarded_ports"
                ]
                current_rewarded_ports = behavior_data["rewarded_ports"]
                other_ports = ~(previous_reward_ports + current_rewarded_ports)

                perseverative_errors[inj].append(np.mean(licks[:, previous_reward_ports]))
                unforgiveable_errors[inj].append(np.mean(licks[:, other_ports]))

        if show_plot:
            fig, axs = plt.subplots(1, 2, sharey=True)
            fig.subplots_adjust(wspace=0)
            ylabel = {True: 'Proportion of trials',
                      False: 'Mean licks per trial'}
            for ax, rate, title in zip(
                    axs,
                    [perseverative_errors, unforgiveable_errors],
                    ["Perseverative errors", "Unforgiveable errors"],
            ):
                boxes = ax.boxplot(
                    [rate[inj] for inj in PSAM_groups], patch_artist=True, widths=0.75,
                    showfliers=False, zorder=0,
                )

                [ax.scatter(
                    jitter_x(np.ones_like(rate[inj])*(i+1), 0.05),
                    rate[inj],
                    color=color,
                    edgecolor='k',
                    zorder=1,
                    s=50,
                )
                    for i, (inj, color) in enumerate(zip(PSAM_groups,
                                                         PSAM_colors))]

                for patch, med, color in zip(
                        boxes["boxes"], boxes["medians"], PSAM_colors
                ):
                    patch.set_facecolor(color)
                    med.set(color="k")
                ax.set_title(title)
                ax.set_xticks([])

            axs[0].set_ylabel(ylabel[binarize])
            patches = [
                mpatches.Patch(facecolor=c, label=label, edgecolor="k")
                for c, label in zip(PSAM_colors, PSAM_groups)
            ]
            fig.legend(handles=patches, loc="lower right")

        return perseverative_errors, unforgiveable_errors

    def scatter_box(self, data, ylabel='', ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        boxes = ax.boxplot([data[inj] for inj in PSAM_groups],
                           widths=0.75, showfliers=False, zorder=0, patch_artist=True)

        [ax.scatter(
            jitter_x(np.ones_like(data[inj])*(i+1), 0.05),
            data[inj],
            color=color,
            edgecolor='k',
            zorder=1,
            s=50,
        )
            for i, (inj, color) in enumerate(zip(PSAM_groups,
                                                 PSAM_colors))]

        for patch, med, color in zip(
                boxes["boxes"], boxes["medians"], PSAM_colors
        ):
            patch.set_facecolor(color)
            med.set(color="k")
        ax.set_xticks([])
        ax.set_ylabel(ylabel)

    def compare_trial_count(self, session_type):
        trials = {inj: [self.data[mouse][session_type].data['ntrials']
                        for mouse in self.meta['grouped_mice'][inj]]
                  for inj in PSAM_groups}

        self.scatter_box(trials, 'Trials')

        return trials

    def compare_licks(self, session_type, exclude_rewarded=False):
        licks = {inj: [] for inj in PSAM_groups}
        for inj in PSAM_groups:
            for mouse in self.meta['grouped_mice'][inj]:
                if exclude_rewarded:
                    ports = ~self.data[mouse][session_type].data['rewarded_ports']
                else:
                    ports = np.ones_like(self.data[mouse][session_type].data['rewarded_ports'],
                                         dtype=bool)
                licks[inj].append(np.sum(self.data[mouse]
                                         [session_type].data['all_licks'][:,ports]))

        self.scatter_box(licks, ylabel='Licks')

        return licks


if __name__ == '__main__':
    mice = ['PSAM_' + str(i) for i in np.arange(6,18)]
    P = PSAM(mice)