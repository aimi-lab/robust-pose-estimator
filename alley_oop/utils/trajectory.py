import json
import os
import numpy as np
from scipy.spatial.transform import Rotation


def save_trajectory(trajectory: list, path: str):
    save_json(trajectory, path)
    save_freiburg(trajectory, path)


def save_json(trajectory: list, path: str):
    with open(os.path.join(path, 'trajectory.json'), 'w') as f:
        json.dump(trajectory, f)


def save_freiburg(trajectory: list, path: str):
    with open(os.path.join(path, 'trajectory.freiburg'), 'w') as f:
        for tr in trajectory:
            rmat = np.asarray(tr['camera-pose'])
            rmat = rmat[:3, :3]
            qs = Rotation.from_matrix(rmat).as_quat()
            t = (tr['camera-pose'][0][3]/1000.0, tr['camera-pose'][1][3]/1000.0, tr['camera-pose'][2][3]/1000.0)

            f.write(f"{tr['timestamp']} {t[0]} {t[1]} {t[2]} {qs[0]} {qs[1]} {qs[2]} {qs[3]}\n")

def json2freiburg(json_path, outpath):
    with open(str(json_path), 'r') as f: pose_elem_list = json.load(f)
    pose_list = []
    for i, pose_elem in enumerate(pose_elem_list):
        pose = np.array(pose_elem['camera-pose'])
        pose[0:3, 3] = -pose[0:3, 3]  # neg. translation as Intuitive's coordinate system is inverted wrt. OpenCV
        pose[0:3, 0:3] = pose[0:3, 0:3].T  # invert rotation as Intuitive's coordinate system is inverted wrt. OpenCV
        pose_list.append({'camera-pose': pose, 'timestamp': pose_elem['timestamp']})
    save_freiburg(pose_list, outpath)
