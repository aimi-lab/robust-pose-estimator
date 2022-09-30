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

def read_freiburg(path: str, ret_stamps=False, no_stamp=False):
    with open(path, 'r') as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
    if no_stamp:
        translation = [l[0:3] for l in list if len(l) > 0]
        quaternions = [l[3:] for l in list if len(l) > 0]
        pose_list = []
        for t, q in zip(translation, quaternions):
            pose = np.eye(4)
            pose[:3, 3] = 1000.0 * np.asarray(t).astype(float)  # m to mm
            pose[:3, :3] = Rotation.from_quat(q).as_matrix()
            pose_list.append(pose)
    else:
        time_stamp = [l[0] for l in list if len(l) > 0]
        try:
            time_stamp = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in time_stamp])*100
        except IndexError:
            time_stamp = np.asarray([int(l) for l in time_stamp])
        translation = [l[1:4] for l in list if len(l) > 0]
        quaternions = [l[4:] for l in list if len(l) > 0]
        pose_list = []
        for t, q in zip(translation, quaternions):
            pose = np.eye(4)
            pose[:3, 3] = 1000.0*np.asarray(t).astype(float)  #m to mm
            pose[:3,:3] = Rotation.from_quat(q).as_matrix()
            pose_list.append(pose)
        if ret_stamps:
            return pose_list, time_stamp
    return pose_list


def read_json_intuitive(path: str, with_stamp=True):
    with open(os.path.join(path), 'r') as f:
        raw = json.load(f)
    poses = []
    if with_stamp:
        time_stamps = np.asarray([r["timestamp"] for r in raw])
    for i, r in enumerate(raw):
        if with_stamp:
            pose = np.eye(4)
            pose[:3,:3] = np.asarray(r["camera_pose"][3:]).reshape(3,3)
            pose[:3,3] = np.asarray(r["camera_pose"][:3])
        else:
            if isinstance(r, dict):
                r = r['camera-pose']
            pose = np.asarray(r)
        poses.append(pose)
    if with_stamp:
        return poses, time_stamps
    return poses
