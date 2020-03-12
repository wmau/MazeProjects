from CircleTrack.BehaviorFunctions import *
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    PlotApproaches(r'D:\Projects\CircleTrack\Mouse4\01_30_2020\H16_M50_S22')
    PlotApproaches(r'D:\Projects\CircleTrack\Mouse4\02_01_2020\H15_M37_S17')

    #PlotBlockedApproaches(r'D:\Projects\CircleTrack\Mouse4\01_30_2020\H16_M50_S22')
    pass