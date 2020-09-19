from CircleTrack.BehaviorFunctions import BehaviorSession
from util import Metadata_CSV


def MultiSession(mouse, Metadata_CSV, behavior='CircleTrack'):
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
        S[session_type] = (BehaviorSession(folder))

    return S


def MultiAnimal(mice, project_folder=r'Z:\Will\Drift\Data',
                behavior='CircleTrack'):
    """
    Gathers all sessions for all specified mice.

    :param project_folder:
    :param behavior:
    :return:
    """
    M = Metadata_CSV(project_folder)

    sessions_by_mouse = dict()
    for mouse in mice:
        sessions_by_mouse[mouse] = MultiSession(mouse, M, behavior=behavior)

    return sessions_by_mouse