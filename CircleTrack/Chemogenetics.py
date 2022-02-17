import numpy as np
from CircleTrack.SessionCollation import MultiAnimal
from CircleTrack.BehaviorFunctions import BehaviorSession
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import ttest_ind
from CaImaging.util import nan_array, sem, stack_padding, group_consecutives
import pandas as pd
from CaImaging.plotting import errorfill, beautify_ax, jitter_x
import matplotlib.patches as mpatches
import os
import pingouin as pg
from statsmodels.stats.multitest import multipletests
from scipy.integrate import simps

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 22})

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
colors = {"PSAM": ["cornflowerblue", "mediumpurple"],
          "DREADDs": ["silver", "coral"]}


class Chemogenetics:
    def __init__(self, mice, actuator='DREADDs', save_figs=True, ext='pdf',
                 save_path=r'Z:\Will\Manuscripts\memory_flexibility\Figures'):
        project_name = project_names[actuator]
        self.data = MultiAnimal(
            mice, project_name=project_name,
            SessionFunction=BehaviorSession
        )

        self.save_configs = {'save_figs': save_figs,
                             'ext': ext,
                             'path': save_path,
                             }
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

    def save_fig(self, fig, fname, folder):
        fpath = os.path.join(self.save_configs['path'], str(folder),
                             f'{fname}.{self.save_configs["ext"]}')
        fig.savefig(fpath)

    def set_legend(self, fig, groups=None, colors=None):
        if groups is None:
            groups = self.meta['groups']
        if colors is None:
            colors = self.meta['colors']

        patches = [
            mpatches.Patch(facecolor=c, label=label, edgecolor="k")
            for c, label in zip(colors, groups)
        ]
        fig.legend(handles=patches, loc="lower right", fontsize=14)

    def behavior_over_trials(self,
                             session_type,
                             window=6,
                             strides=2,
                             performance_metric='CRs',
                             trial_limit=None):
        dv, mice, sessions, groups, trial_blocks = [], [], [], [], []
        for group in self.meta['groups']:
            for mouse in self.meta['grouped_mice'][group]:
                session = self.data[mouse][session_type]
                session.sdt_trials(
                    rolling_window=window,
                    trial_interval=strides,
                    plot=False,
                    trial_limit=trial_limit,
                )

                n = range(len(session.sdt[performance_metric]))
                dv.extend(session.sdt[performance_metric])
                mice.extend([mouse for i in n])
                sessions.extend([session_type for i in n])
                groups.extend([group for i in n])
                trial_blocks.extend([i for i in n])

        df = pd.DataFrame(
            {
                't': trial_blocks,
                'dv': dv,
                'mice': mice,
                'session': sessions,
                'group': groups,
            }
        )

        # cols = ['mice', 'session', 'group']
        # df[cols] = df[cols].mask(df[cols]=='nan', None).ffill(axis=0)
        # df['t'] = df['t'].isna().cumsum() + df['t'].ffill()
        #df = df.dropna(axis=0)

        return df

    def stack_behavior_dv(self, df):
        dv = dict()

        for group in self.meta['groups']:
            dv_temp = []

            for mouse in self.meta['grouped_mice'][group]:
                dv_temp.append(df.loc[df['mice']==mouse, 'dv'])
                dv[group] = stack_padding(dv_temp)

        return dv

    def plot_trial_behavior(self, session_types=None, performance_metric='d_prime',
                            plot_sig=False, **kwargs):
        if session_types is None:
            session_types = self.meta['session_types']
        elif type(session_types) is str:
            session_types = [session_types]
        n_sessions = len(session_types)

        dv, pvals, anova_dfs = dict(), dict(), dict()
        for session_type in session_types:
            anova_dfs[session_type], df, pvals[session_type] =\
                self.trial_behavior_anova(session_type, performance_metric=performance_metric,
                                          **kwargs)
            dv[session_type] = self.stack_behavior_dv(df)

        ylabel = {
            'd_prime': "d'",
            'CRs': "Correct rejection rate",
            'hits': "Hit rate",
        }
        if n_sessions == 1:
            fig, axs = plt.subplots(1,n_sessions, figsize=(5,5))
            axs = [axs]
        else:
            fig, axs = plt.subplots(1, n_sessions, figsize=(5*n_sessions, 5),
                                    sharey=True)
        for i, (ax, session_type) in enumerate(zip(axs, session_types)):
            for group, color in zip(self.meta['groups'], self.meta['colors']):
                y = dv[session_type][group]
                xrange = y.shape[1]
                ax.plot(range(xrange), y.T, color=color, alpha=0.3)
                errorfill(
                    range(xrange),
                    np.nanmean(y, axis=0),
                    sem(y, axis=0),
                    ax=ax,
                    color=color,
                    label=group
                )
            xlims = [int(x) for x in ax.get_xlim()]
            ax.set_xticks(xlims)
            ax.set_title(session_type.replace('Goals', 'Training'))
            ax.set_xticklabels([1, xlims[-1]])

            if i == 0:
                ax.set_ylabel(ylabel[performance_metric])
            [ax.spines[side].set_visible(False) for side in ['top', 'right']]

            if plot_sig:
                sig = np.where(pvals[session_type] < 0.05)[0]
                if len(sig):
                    sig_regions = group_consecutives(sig, step=1)
                    ylims = ax.get_ylim()
                    for region in sig_regions:
                        ax.fill_between(np.arange(region[0], region[-1]), ylims[-1], ylims[0], alpha=0.4, color='gray')

        if n_sessions==1:
            axs[0].set_xlabel('Sliding trial windows')
        else:
            fig.supxlabel("Sliding trial windows")
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        axs[-1].legend(loc='lower right', fontsize=14)

        return dv, anova_dfs, fig

    def trial_behavior_anova(self, session_type, performance_metric='d_prime', **kwargs):
        df = self.behavior_over_trials(session_type, performance_metric=performance_metric,
                                       **kwargs)

        anova_df = pg.anova(df, dv='dv', between=['t', 'group'], ss_type=1)

        pvals = []
        for i in range(np.max(df['t'])):
            x = df.loc[np.logical_and(df['group']==self.meta['groups'][0], df['t']==i), 'dv']
            y = df.loc[np.logical_and(df['group']==self.meta['groups'][1], df['t']==i), 'dv']

            pval = ttest_ind(x,y, nan_policy='omit').pvalue
            if np.isfinite(pval) and ((len(x) + len(y)) > 5):
                pvals.append(pval)

        pvals = multipletests(pvals, method='fdr_bh')[1]

        return anova_df, df, pvals

    # def get_learning_rates(self, dv):
    #     learning_rates = dict()
    #     for group in self.meta['groups']:
    #         learning_rates[group] = []
    #
    #         for performance in dv[group]:
    #             performance = performance[np.isfinite(performance)]
    #             learning_rates[group].append(spearmanr(np.arange(len(performance)),
    #                                                              performance).correlation)
    #
    #     return learning_rates

    def plot_reversal_vs_training4_trial_behavior(self, group, performance_metric='d_prime',
                                                  **kwargs):
        dv, pvals = dict(), dict()
        reversal_color = {
            'vehicle': 'cornflowerblue',
            'PSEM': self.meta['colors'][self.meta['groups'].index(group)],
        }
        session_types = ('Goals4', 'Reversal')
        for session_type in session_types:
            df = self.trial_behavior_anova(session_type, performance_metric=performance_metric,
                                           **kwargs)[1]
            dv[session_type] = self.stack_behavior_dv(df)

        # For significance markers.
        pvals = []
        for x, y in zip(dv[session_types[0]][group].T, dv[session_types[1]][group].T):
            pval = ttest_ind(x,y, nan_policy='omit').pvalue

            if np.isfinite(pval) and ((len(x) + len(y)) > 5):
                pvals.append(pval)
        pvals = multipletests(pvals, method='fdr_bh')[1]

        ylabel = {
            'd_prime': "d'",
            'CRs': "Correct rejection rate",
            'hits': "Hit rate",
        }
        fig, ax = plt.subplots(figsize=(5,5))
        for session_type, color in zip(session_types, ['k', reversal_color[group]]):
            y = dv[session_type][group]
            x = y.shape[1]
            ax.plot(range(x), y.T, color=color, alpha=0.3)
            errorfill(range(x),
                      np.nanmean(y, axis=0),
                      sem(y, axis=0),
                      ax=ax,
                      color=color,
                      label=session_type.replace('Goals', 'Training'))

        sig = np.where(pvals < 0.05)[0]
        if len(sig):
            sig_regions = group_consecutives(sig, step=1)
            ylims = ax.get_ylim()
            for region in sig_regions:
                ax.fill_between(np.arange(region[0], region[-1]), ylims[-1], ylims[0],
                                alpha=0.4, color='gray')
        ax.set_title(group, color=reversal_color[group])
        ax.legend(loc='lower right', fontsize=14)
        ax.set_ylabel(ylabel[performance_metric])
        ax.set_xlabel('Sliding trial windows')
        [ax.spines[side].set_visible(False) for side in ['top', 'right']]
        fig.tight_layout()

        return dv, fig

    def aggregate_behavior_over_trials(
        self,
        window=6,
        strides=2,
        performance_metric="d_prime",
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
        strides=None,
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

        behavioral_performance = self.aggregate_behavior_over_trials(
            window=window,
            strides=strides,
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

            self.scatter_box(best_performance, ax=ax)

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
        strides=None,
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
                strides=strides,
                performance_metric=performance_metric,
                downsample_trials=downsample_trials,
            )
            ax.set_xticks([])
            ax.set_title(title, fontsize=16)

            [ax.spines[side].set_visible(False) for side in ['top', 'right']]

        axs[0].set_ylabel(ylabels[performance_metric])

        fig.tight_layout()
        fig.subplots_adjust(wspace=0)
        self.set_legend(fig)

        df = pd.concat(
            [
                pd.concat({k: pd.Series(v) for k, v in performance[session].items()})
                for session in sessions
            ],
            axis=1,
            keys=sessions,
        )

        if self.save_configs['save_figs']:
            self.save_fig(fig, f'session {performance_metric}', 1)

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
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(performance_all.T, color="k", alpha=0.5)
        errorfill(
            session_labels,
            np.mean(performance_all, axis=0),
            sem(performance_all, axis=0),
            ax=ax,
            color="k",
        )
        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        [ax.spines[side].set_visible(False) for side in ['top', 'right']]
        ax.set_ylabel(ylabel[performance_metric])
        fig.tight_layout()

        df = pd.DataFrame(
            performance_all,
            index=np.hstack(
                [mice for mice in self.meta['grouped_mice'].values()]
            ),
            columns=sessions,
        )

        if self.save_configs['save_figs']:
            self.save_fig(fig, f'all_mice_{performance_metric}', 1)

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

    def plot_perseverative_licking_over_session(
                self,
                session_type="Reversal",
                window_size=6,
                trial_interval=2,
                show_plot=True,
                binarize_licks=True,
        ):
        """
        Plot perseverative and unforgiveable errors averaged across trial windows


        """
        perseverative_errors = dict()
        unforgiveable_errors = dict()
        for group in self.meta['groups']:
            perseverative_errors[group] = [[] for mouse in self.meta["grouped_mice"][group]]
            unforgiveable_errors[group] = [[] for mouse in self.meta["grouped_mice"][group]]

            for i, mouse in enumerate(self.meta["grouped_mice"][group]):
                behavior = self.data[mouse][session_type]
                behavior_data = behavior.data

                licks = behavior.rolling_window_licks(window_size, trial_interval)
                if binarize_licks:
                    licks = licks > 0

                # Find previously rewarded, currently rewarded, and never
                # rewarded ports.
                previous_reward_ports = self.data[mouse]["Goals4"].data[
                    "rewarded_ports"
                ]
                current_rewarded_ports = behavior_data["rewarded_ports"]
                other_ports = ~(previous_reward_ports + current_rewarded_ports)
                n_previous = np.sum(previous_reward_ports)
                n_other = np.sum(other_ports)

                for licks_this_window in licks:
                    # Get perserverative errors.
                    perseverative_errors[group][i].append(
                        np.sum(licks_this_window[:, previous_reward_ports])
                        / (n_previous * licks_this_window.shape[0])
                    )

                    # Get unforgiveable errors.
                    unforgiveable_errors[group][i].append(
                        np.sum(licks_this_window[:, other_ports])
                        / (n_other * licks_this_window.shape[0])
                    )

        perseverative_errors = {
            group: stack_padding(perseverative_errors[group]) for group in self.meta['groups']
        }
        unforgiveable_errors = {
            group: stack_padding(unforgiveable_errors[group]) for group in self.meta['groups']
        }

        if show_plot:
            ylabel = "Error rate" if binarize_licks else "Average number of licks"
            if session_type == "Reversal":
                fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
                fig.subplots_adjust(wspace=0)

                for ax, rate, title in zip(
                        axs,
                        [perseverative_errors, unforgiveable_errors],
                        ["Perseverative errors", "Unforgiveable errors"],
                ):
                    se = {group: sem(rate[group], axis=0) for group in self.meta['groups']}
                    m = {group: np.nanmean(rate[group], axis=0) for group in self.meta['groups']}
                    for c, group in zip(self.meta['colors'], self.meta['groups']):
                        ax.plot(rate[group].T, color=c, alpha=0.1)
                        errorfill(
                            range(m[group].shape[0]), m[group], se[group], color=c, ax=ax
                        )
                        ax.set_title(title)
                    fig.supxlabel("Trial blocks")
                    fig.supylabel(ylabel)
            else:
                fig, ax = plt.subplots()
                se = {group: sem(unforgiveable_errors[group], axis=0) for group in self.meta['groups']}
                m = {group: np.nanmean(unforgiveable_errors[group], axis=0) for group in self.meta['groups']}
                for c, group in zip(self.meta['colors'], self.meta['groups']):
                    ax.plot(unforgiveable_errors[group].T, color=c, alpha=0.1)
                    errorfill(range(m[group].shape[0]), m[group], se[group], color=c, ax=ax)
                ax.set_xlabel("Trial blocks")
                ax.set_ylabel(ylabel)

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
                s=100,
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

    def make_fig1(self, panels=None):
        if panels is None:
            panels = ['C', 'D', 'E', 'F']

        if 'C' in panels:
            d_prime = self.plot_behavior_grouped(performance_metric='d_prime',
                                              sessions=['Goals' + str(i) for i in np.arange(1,5)])
            anova_results = pg.rm_anova(d_prime)
            print(anova_results)

        if 'D' in panels:
            performance_metric = 'CRs'
            dv, anova_dfs, fig = \
                self.plot_trial_behavior(session_types=['Goals4','Reversal'],
                                         performance_metric=performance_metric,
                                         window=6, strides=2)

            if self.save_configs['save_figs']:
                self.save_fig(fig, f'Vehicle vs PSEM Reversal_{performance_metric}', 1)

            for df in anova_dfs.values():
                print(df)


if __name__ == "__main__":
    PSAM_mice = ['PSAM_' + str(i) for i in np.arange(4,28)]
    mistargets = ['PSAM_' + str(i) for i in [4, 5, 6, 8]]
    PSAM_mice.remove('PSAM_18') # Bad reversal session.
    exclude_non_learners = True
    if exclude_non_learners:
        [PSAM_mice.remove(x) for x in ['PSAM_' + str(i) for i in [13,15]]]
    [PSAM_mice.remove(x) for x in mistargets]
    C = Chemogenetics(PSAM_mice, actuator='PSAM')