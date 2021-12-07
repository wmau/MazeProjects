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

project_names = {"DREADDs": "DREADDs_Reversal",
                 "PSAM": "PSAMReversal"}

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
    "DREADDs_Reversal": [
        "Training1",
        "Training2",
        "Training3",
        "Training4",
        "Reversal",
    ],
    "PSAMReversal": ["Goals1", "Goals2", "Goals3", "Goals4", "Reversal"],
}

# Exclude PSAM4, PSAM5, and PSAM15 because they never learned
# (poor Goals4 performance).
grouped_mice = {
    "DREADDs": ["DREADDs_2", "DREADDs_4", "DREADDs_5", "DREADDs_7"],
    "PSAM": [
        "PSAM_2",
        "PSAM_3",
        "PSAM_5",
        "PSAM_6",
        "PSAM_7",
        "PSAM_8",
        "PSAM_10",
        "PSAM_13",
        "PSAM_17",
        "PSAM_19",
        "PSAM_20",
        "PSAM_24",
        "PSAM_25",
        "PSAM_26",
        "PSAM_27",
    ],
}

groups = {"DREADDs": ["fluorophore", "hM4di"],
          "PSAM": ["vehicle", "PSEM"]}
age_colors = ["cornflowerblue", "r"]
colors = {"PSAM": ["silver", "mediumpurple"],
          "DREADDs": ["silver", "coral"]}


class Chemogenetics:
    def __init__(self, mice, actuator='DREADDs'):
        project_name = project_names[actuator]
        self.data = MultiAnimal(
            mice, project_name=project_name,
            SessionFunction=BehaviorSession
        )

        self.meta = {
            "session_types": session_types[project_name],
            "mice": mice,
            "groups": groups[actuator],
            "colors": colors[actuator],
        }

        self.meta["session_labels"] = [
            session_type.replace("Goals", "Training")
            for session_type in self.meta["session_types"]
        ]

        self.meta["grouped_mice"] = {key: [mouse for mouse in self.meta['mice'] if mouse in grouped_mice[actuator]]
                                     for key in groups[actuator]}

        self.meta["grouped_mice"] = dict()
        for key in groups[actuator]:
            if key in ["hM4di", "PSEM"]:
                mouse_list = [mouse for mouse in self.meta["mice"] if mouse in grouped_mice[actuator]]
            else:
                mouse_list = [mouse for mouse in self.meta["mice"] if mouse not in grouped_mice[actuator]]
            self.meta["grouped_mice"][key] = mouse_list

    def set_legend(self, fig, groups=None, colors=None):
        if groups is None:
            groups = self.meta['groups']
        if colors is None:
            colors = self.meta['colors']

        patches = [
            mpatches.Patch(facecolor=c, label=label, edgecolor="k")
            for c, label in zip(colors, groups)
        ]
        fig.legend(handles=patches, loc="lower right")

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
            group: (len(self.meta["grouped_mice"][group]), sum(longest_sessions))
            for group in self.meta['groups']
        }
        metrics = {key: nan_array(dims[key]) for key in self.meta['groups']}

        for group in self.meta['groups']:
            for row, mouse in enumerate(self.meta["grouped_mice"][group]):
                for border, session in zip(borders, self.meta["session_types"]):
                    metric_this_session = behavioral_performance.sel(
                        metric=performance_metric, mouse=mouse, session=session
                    ).values.tolist()
                    length = len(metric_this_session)
                    metrics[group][row, border : border + length] = metric_this_session

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
                for group, c in zip(self.meta['groups'], self.meta['colors']):
                    ax.plot(metrics[group].T, color=c, alpha=0.3)
                    errorfill(
                        range(metrics[group].shape[1]),
                        np.nanmean(metrics[group], axis=0),
                        sem(metrics[group], axis=0),
                        ax=ax,
                        color=c,
                        label=group,
                    )

                for session in borders[1:]:
                    ax.axvline(x=session, color="k")
                ax.set_xticks(borders)
                ax.set_xticklabels(np.insert(longest_sessions, 0, 0))
                ax.set_xlabel("Trial blocks")
                _ = beautify_ax(ax)
            else:
                for group, c in zip(self.meta['groups'], self.meta['colors']):
                    ax.plot(metrics[group].T, color=c, alpha=0.3)
                    ax.errorbar(
                        self.meta["session_labels"],
                        np.nanmean(metrics[group], axis=0),
                        sem(metrics[group], axis=0),
                        color=c,
                        label=group,
                        capsize=5,
                        linewidth=3,
                    )
            ax.set_ylabel(ylabels[performance_metric])
            fig.legend()

        if window is None:
            mice_ = np.hstack(
                [
                    np.repeat(
                        self.meta["grouped_mice"][group], len(self.meta["session_types"])
                    )
                    for group in self.meta['groups']
                ]
            )
            groups_ = np.hstack(
                [np.repeat(group, metrics[group].size) for group in self.meta['groups']]
            )
            session_types_ = np.hstack(
                [
                    np.tile(
                        self.meta["session_types"], len(self.meta["grouped_mice"][group])
                    )
                    for group in self.meta['groups']
                ]
            )
            metric_ = np.hstack([metrics[inj].flatten() for inj in self.meta['groups']])

            df = pd.DataFrame(
                {
                    "metric": metric_,
                    "session_types": session_types_,
                    "mice": mice_,
                    "group": groups_,
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
        for group in self.meta['groups']:
            best_performance[group] = []
            for mouse in self.meta["grouped_mice"][group]:
                best_performance[group].append(
                    np.nanmax(
                        behavioral_performance.sel(
                            mouse=mouse, metric=performance_metric, session=session_type
                        ).values.tolist()
                    )
                )

        if show_plot:
            label_axes = True
            ylabels = {
                "CRs": "Correct rejection rate",
                "hits": "Hit rate",
                "d_prime": "d'",
            }
            if ax is None:
                fig, ax = plt.subplots(figsize=(3, 4.75))
            else:
                label_axes = False
            box = ax.boxplot(
                [best_performance[inj] for inj in self.meta['groups']],
                labels=self.meta['groups'],
                patch_artist=True,
                widths=0.75,
                showfliers=False,
                zorder=0,
            )

            [
                ax.scatter(
                    jitter_x(np.ones_like(best_performance[group]) * (i + 1), 0.1),
                    best_performance[group],
                    color=color,
                    edgecolor="k",
                    zorder=1,
                )
                for i, (group, color) in enumerate(zip(self.meta['groups'], self.meta['colors']))
            ]
            for patch, med, color in zip(box["boxes"], box["medians"], self.meta['colors']):
                patch.set_facecolor(color)
                med.set(color="k")

            if label_axes:
                ax.set_xticks([1, 2])
                ax.set_xticklabels(self.meta['groups'])
                ax.set_ylabel(ylabels[performance_metric])
                # ax = beautify_ax(ax)
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
            sessions = self.meta["session_types"]

        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in sessions
        ]
        fig, axs = plt.subplots(1, len(sessions), sharey=True)
        fig.subplots_adjust(wspace=0)
        ylabels = {
            "d_prime": "d'",
            "CRs": "Correct rejection rate",
            "hits": "Hit rate",
        }
        performance = dict()
        for ax, session, title in zip(axs, sessions, session_labels):
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

        self.set_legend(fig)

        df = pd.concat(
            [
                pd.concat({k: pd.Series(v) for k, v in performance[session].items()})
                for session in sessions
            ],
            axis=1,
            keys=sessions,
        )

        return performance, df

    def plot_behavior_grouped(
        self, window=None, performance_metric="d_prime", sessions=None
    ):
        if sessions is None:
            sessions = self.meta["session_types"]

        session_labels = [
            self.meta["session_labels"][self.meta["session_types"].index(session)]
            for session in sessions
        ]

        performance_all = nan_array((len(self.meta["mice"]), len(sessions)))

        for i, session in enumerate(sessions):
            performance = self.plot_best_performance(
                session_type=session,
                window=window,
                performance_metric=performance_metric,
                show_plot=False,
            )

            performance[self.meta['groups'][0]].extend(performance[self.meta['groups'][1]])
            performance_all[:, i] = performance[self.meta['groups'][0]]

        ylabel = {"CRs": "Correct rejection rate", "hits": "Hit rate", "d_prime": "d'"}
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.plot(performance_all.T, color="gray", alpha=0.5)
        errorfill(
            session_labels,
            np.mean(performance_all, axis=0),
            sem(performance_all, axis=0),
            ax=ax,
            color="k",
        )
        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        ax.set_ylabel(ylabel[performance_metric])
        fig.tight_layout()

        df = pd.DataFrame(
            performance_all,
            index=np.hstack(
                [mice for mice in self.meta['grouped_mice'].values()]
            ),
            columns=sessions,
        )

        return df

    def plot_perseverative_licking(self, show_plot=True, binarize=True):
        goals4 = self.meta['session_types'][-2]
        reversal = "Reversal"

        perseverative_errors = dict()
        unforgiveable_errors = dict()
        for group in self.meta['groups']:
            perseverative_errors[group] = []
            unforgiveable_errors[group] = []

            for mouse in self.meta["grouped_mice"][group]:
                behavior_data = self.data[mouse][reversal].data

                if binarize:
                    licks = behavior_data["all_licks"] > 0
                else:
                    licks = behavior_data["all_licks"]

                previous_reward_ports = self.data[mouse][goals4].data["rewarded_ports"]
                current_rewarded_ports = behavior_data["rewarded_ports"]
                other_ports = ~(previous_reward_ports + current_rewarded_ports)

                perseverative_errors[group].append(
                    np.mean(licks[:, previous_reward_ports])
                )
                unforgiveable_errors[group].append(np.mean(licks[:, other_ports]))

        if show_plot:
            fig, axs = plt.subplots(1, 2, sharey=True)
            fig.subplots_adjust(wspace=0)
            ylabel = {True: "Proportion of trials", False: "Mean licks per trial"}
            for ax, rate, title in zip(
                axs,
                [perseverative_errors, unforgiveable_errors],
                ["Perseverative errors", "Unforgiveable errors"],
            ):
                boxes = ax.boxplot(
                    [rate[inj] for inj in self.meta['groups']],
                    patch_artist=True,
                    widths=0.75,
                    showfliers=False,
                    zorder=0,
                )

                [
                    ax.scatter(
                        jitter_x(np.ones_like(rate[group]) * (i + 1), 0.05),
                        rate[group],
                        color=color,
                        edgecolor="k",
                        zorder=1,
                        s=50,
                    )
                    for i, (group, color) in enumerate(
                        zip(self.meta['groups'], self.meta['colors'])
                    )
                ]

                for patch, med, color in zip(
                    boxes["boxes"], boxes["medians"], self.meta['colors']
                ):
                    patch.set_facecolor(color)
                    med.set(color="k")
                ax.set_title(title)
                ax.set_xticks([])

            axs[0].set_ylabel(ylabel[binarize])
            self.set_legend(fig)

        return perseverative_errors, unforgiveable_errors

    def scatter_box(self, data, ylabel="", ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        boxes = ax.boxplot(
            [data[inj] for inj in self.meta['groups']],
            widths=0.75,
            showfliers=False,
            zorder=0,
            patch_artist=True,
        )

        [
            ax.scatter(
                jitter_x(np.ones_like(data[group]) * (i + 1), 0.05),
                data[group],
                color=color,
                edgecolor="k",
                zorder=1,
                s=50,
            )
            for i, (group, color) in enumerate(zip(self.meta['groups'], self.meta['colors']))
        ]

        for patch, med, color in zip(boxes["boxes"], boxes["medians"], self.meta['colors']):
            patch.set_facecolor(color)
            med.set(color="k")
        ax.set_xticks([])
        ax.set_ylabel(ylabel)

    def compare_trial_count(self, session_type):
        trials = {
            inj: [
                self.data[mouse][session_type].data["ntrials"]
                for mouse in self.meta["grouped_mice"][inj]
            ]
            for inj in self.meta['groups']
        }

        self.scatter_box(trials, "Trials")

        return trials

    def compare_licks(self, session_type, exclude_rewarded=False):
        licks = {group: [] for group in self.meta['groups']}
        for group in self.meta['groups']:
            for mouse in self.meta["grouped_mice"][group]:
                if exclude_rewarded:
                    ports = ~self.data[mouse][session_type].data["rewarded_ports"]
                else:
                    ports = np.ones_like(
                        self.data[mouse][session_type].data["rewarded_ports"],
                        dtype=bool,
                    )
                licks[group].append(
                    np.sum(self.data[mouse][session_type].data["all_licks"][:, ports])
                )

        self.scatter_box(licks, ylabel="Licks")

        return licks


if __name__ == "__main__":
    mice = ["PSAM_" + str(i) for i in np.arange(6, 18)]
    P = Chemogenetics(mice)
