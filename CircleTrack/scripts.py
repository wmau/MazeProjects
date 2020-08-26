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
        self.all_sessions = MultiAnimal(mice, project_folder, behavior='CircleTrack')
        for animal, sessions in self.all_sessions:
            for session in sessions:
                session.get_licks(plot=False)


        pass


    #def arrange_licking(self):





if __name__ == '__main__':
    B = BatchBehaviorAnalyses(['Betelgeuse_Scope25', 'Alcor_Scope20', 'M1', 'M2', 'M3', 'M4'])