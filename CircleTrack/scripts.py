from CircleTrack.BehaviorFunctions import *
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False
plt.rcParams.update({'font.size': 12})
from util import Metadata_CSV

project_folder = r'Z:\Will\Drift\Data'
M = Metadata_CSV(project_folder)

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


def Sessions_by_Mouse(mouse, behavior='CircleTrack'):
    """
    Gathers session data for one mouse.

    :param mouse:
    :param behavior:
    :return:
    """
    # Find the folders corresponding to the correct mouse and behavior.
    mouse_entries = M.df.loc[M.df['Mouse'] == mouse]
    sessions = mouse_entries.loc[mouse_entries['Session'].str.find(behavior) > 0]

    S = []
    for folder in sessions['Path']:
        S.append(Session(folder))

    return S




if __name__ == '__main__':
    mouse = 'Alcor_Scope20'
    Sessions_by_Mouse(mouse, behavior='CircleTrack')