from CircleTrack.BehaviorFunctions import BehaviorSession
from CircleTrack.MiniscopeFunctions import CalciumSession
from util import Metadata_CSV
import numpy as np
from CaImaging.CellReg import CellRegObj

def MultiSession(mouse, Metadata_CSV, behavior='CircleTrack',
                 SessionFunction=BehaviorSession):
    """
    Gathers session data for one mouse.

    :param mouse:
    :param behavior:
    :return:
    """
    # Find the folders corresponding to the correct mouse and behavior.
    mouse_entries = Metadata_CSV.df.loc[Metadata_CSV.df['Mouse'] == mouse]
    sessions = mouse_entries.loc[mouse_entries['Session'].str.find(behavior) > 0]

    S = dict()
    for folder, session_type in zip(sessions['Path'], sessions['Session_Type']):
        S[session_type] = (SessionFunction(folder))

    if SessionFunction==CalciumSession:
        cellreg_path = np.unique(mouse_entries['CellRegPath'])[0]
        try:
            S['CellReg'] = CellRegObj(cellreg_path)
        except:
            print(f'CellReg for {mouse} failed to load.')

    return S


def MultiAnimal(mice, project_folder=r'Z:\Will\Drift\Data',
                behavior='CircleTrack',
                SessionFunction=BehaviorSession):
    """
    Gathers all sessions for all specified mice.

    :param project_folder:
    :param behavior:
    :return:
    """
    M = Metadata_CSV(project_folder)

    sessions_by_mouse = dict()
    for mouse in mice:
        sessions_by_mouse[mouse] =\
            MultiSession(mouse, M, behavior=behavior,
                         SessionFunction=SessionFunction)

    return sessions_by_mouse