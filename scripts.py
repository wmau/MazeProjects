from CircleTrack.BehaviorFunctions import *
import matplotlib.pyplot as plt

def PlotApproaches(folder, accleration=True):
    """
    Plot approach speeds to each water port.

    :param folder:
    :param accleration:
    :return:
    """
    data = Process(folder)
    fig = plt.figure(figsize=(6,8.5))

    # Get approach speeds (or acceleration).
    approaches = []
    for i in range(8):
        ax = fig.add_subplot(4,2,i+1)
        approach = approach_speed(data.behavior_df, data.lin_ports[i], ax=ax,
                                  acceleration=accleration)

        approaches.append(approach)

    # Normalize color axes across all subplots.
    fig.tight_layout()
    max_speed = np.max([np.nanmax(approach) for approach in approaches])
    min_speed = np.min([np.nanmin(approach) for approach in approaches])
    for ax in fig.axes:
        for im in ax.get_images():
            im.set_clim(min_speed, max_speed)

    fig.show()


def PlotBlockedApproaches(folder, acceleration=True, blocks=4):
    data = Process(folder)
    fig = plt.figure(figsize=(6, 8.5))
    for i in range(8):
        try:
            ax = fig.add_subplot(4, 2, i + 1, sharey=ax)
        except:
            ax = fig.add_subplot(4, 2, i + 1)
        approaches = approach_speed(data.behavior_df, data.lin_ports[i], plot=False,
                                    smoothing_factor=3, acceleration=acceleration)

        blocked_approach_speeds(approaches, blocks=blocks, ax=ax)
    fig.tight_layout()
    fig.show()

if __name__ == '__main__':
    PlotApproaches(r'D:\Projects\CircleTrack\Mouse4\01_30_2020\H16_M50_S22')
    PlotApproaches(r'D:\Projects\CircleTrack\Mouse4\02_01_2020\H15_M37_S17')

    PlotBlockedApproaches(r'D:\Projects\CircleTrack\Mouse4\01_30_2020\H16_M50_S22')
    pass