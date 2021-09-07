from CircleTrack.BehaviorFunctions import BehaviorSession
from CircleTrack.MiniscopeFunctions import CalciumSession
from LinearTrack.BehaviorFunctions import BehaviorSession as LTBehaviorSession
from CaImaging.CellReg import CellRegObj
from CircleTrack.sql import Database
import os

directory = r'D:'
db_fname = 'database.sqlite'

def MultiSession(mouse, project_name='Drift',
                 directory=directory, db_fname=db_fname,
                 SessionFunction=CalciumSession,
                 session_types=None,
                 **kwargs):

    db = Database(directory, db_fname)
    sql_str = """
        SELECT session.session_name, session.path
        FROM session
        INNER JOIN mouse
        ON mouse.id = session.mouse_id
        INNER JOIN project
        ON project.id = session.project_id
        WHERE name = ? AND project_name = ?
    """
    results = db.execute(sql_str, (mouse, project_name))

    if session_types is None:
        session_types = [r[0] for r in results]

    S = dict()
    for session_type, folder in results:
        if session_type in session_types:
            S[session_type] = SessionFunction(folder, **kwargs)

    sql_str = """
        SELECT project.path
        FROM project
        INNER JOIN session
        ON session.project_id = project.id
        WHERE project.project_name = ?
        LIMIT 1
    """
    results = db.execute(sql_str, (project_name,))

    if SessionFunction == CalciumSession:
        cellreg_path = os.path.join(results[0][0], mouse, 'SpatialFootprints', 'CellRegResults')
        try:
            S["CellReg"] = CellRegObj(cellreg_path)
        except:
            print(f"CellReg for {mouse} failed to load.")

    return S


def MultiAnimal(mice, project_name='Drift',
                SessionFunction=CalciumSession,
                session_types=None):
    sessions_by_mouse = dict()

    for mouse in mice:
        print(f"Loading data from {mouse}")
        sessions_by_mouse[mouse] = MultiSession(
            mouse, project_name=project_name,
            SessionFunction=SessionFunction,
            session_types=session_types,
        )

    return sessions_by_mouse


if __name__ == '__main__':
    MultiSession('Io', 'Drift')