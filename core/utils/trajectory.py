import json
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from lietorch import SE3


def mat2SE3(transforms: torch.tensor):
    assert transforms.shape[-2:] == (4,4)
    quat = Rotation.from_matrix(transforms[..., :3, :3]).as_quat()
    trans = transforms[..., :3, 3]
    pose_data = np.concatenate((trans, quat), axis=-1)
    return SE3.InitFromVec(torch.tensor(pose_data))


def save_trajectory(trajectory: list, path: str):
    with open(os.path.join(path, 'trajectory.freiburg'), 'w') as f:
        for tr in trajectory:
            assert isinstance(tr['camera-pose'], SE3)
            vec = tr['camera-pose'].vec().cpu().squeeze().numpy()
            t = (vec[0]/1000.0, vec[1]/1000.0, vec[2]/1000.0)
            f.write(f"{tr['timestamp']} {t[0]} {t[1]} {t[2]} {vec[3]} {vec[4]} {vec[5]} {vec[6]}\n")


def json2freiburg(json_path, outpath):
    with open(str(json_path), 'r') as f: pose_elem_list = json.load(f)
    pose_list = []
    for i, pose_elem in enumerate(pose_elem_list):
        pose = np.array(pose_elem['camera-pose'])
        pose[0:3, 3] = -pose[0:3, 3]  # neg. translation as Intuitive's coordinate system is inverted wrt. OpenCV
        pose[0:3, 0:3] = pose[0:3, 0:3].T  # invert rotation as Intuitive's coordinate system is inverted wrt. OpenCV
        pose_se3 = mat2SE3(torch.tensor(pose).unsqueeze(0)).squeeze()
        pose_list.append({'camera-pose': pose_se3, 'timestamp': pose_elem['timestamp']})
    save_trajectory(pose_list, outpath)


def read_freiburg(path: str, ret_stamps=False, no_stamp=False):
    with open(path, 'r') as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
    if no_stamp:
        trans = torch.from_numpy(np.asarray([l[0:3] for l in list if len(l) > 0], dtype=float))
        trans *= 1000.0  #m to mm
        quat = torch.from_numpy(np.asarray([l[3:] for l in list if len(l) > 0], dtype=float))
        pose_se3 = SE3.InitFromVec(torch.cat((trans, quat), dim=-1))
    else:
        time_stamp = [l[0] for l in list if len(l) > 0]
        try:
            time_stamp = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in time_stamp])*100
        except IndexError:
            time_stamp = np.asarray([int(l) for l in time_stamp])
        trans = torch.from_numpy(np.asarray([l[1:4] for l in list if len(l) > 0], dtype=float))
        trans *= 1000.0  # m to mm
        quat = torch.from_numpy(np.asarray([l[4:] for l in list if len(l) > 0], dtype=float))
        pose_se3 = SE3.InitFromVec(torch.cat((trans, quat), dim=-1))
        if ret_stamps:
            return pose_se3, time_stamp
    return pose_se3


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
        poses.append(torch.tensor(pose))
    poses = mat2SE3(torch.cat(poses, dim=0))
    if with_stamp:
        return poses, time_stamps
    return poses
