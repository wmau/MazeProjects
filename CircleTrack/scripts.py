from CircleTrack.BehaviorFunctions import *
import matplotlib.pyplot as plt

from CircleTrack.BehaviorFunctions import MultiSession

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


if __name__ == '__main__':
    mouse = 'Alcor_Scope20'
    MultiSession(mouse, behavior='CircleTrack')