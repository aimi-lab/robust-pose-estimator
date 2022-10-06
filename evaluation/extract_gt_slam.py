import os
import yaml
import numpy as np
import glob
import json
from alley_oop.utils.trajectory import save_freiburg, read_json_intuitive
import shutil

PATH_REPLACEMENT = ['/home/mhayoz/research', '/storage/workspaces/artorg_aimi/ws_00000']
if __name__ == '__main__':

    with open(os.path.join('..', 'intuitive_slam.txt'), 'r') as f:
        sequences = f.readlines()
    for sequence in sequences:
        print(sequence)
        sequence = sequence.replace(PATH_REPLACEMENT[1], PATH_REPLACEMENT[0])
        sequence = sequence.replace('\n', '')
        file = glob.glob(os.path.join(sequence, "*.json"))
        assert len(file) == 1
        # read poses
        poses = read_json_intuitive(file[0], with_stamp=False)
        # seems to be in meters -> transform to mm
        poses = np.stack(poses)
        poses[:, :3,3] *= 1000
        first_pose = np.linalg.inv(poses[0])
        poses = first_pose[None,...] @ poses

        isi_t = np.eye(4)
        isi_t[0,0] = -1
        isi_t[1,1] = -1
        poses = isi_t[None,...] @ poses @ isi_t[None,...]
        poses = [{"camera-pose": p, "timestamp":i} for i,p in enumerate(poses)]
        save_freiburg(poses, sequence)
        shutil.move(os.path.join(sequence, 'trajectory.freiburg'), os.path.join(sequence, 'groundtruth.txt'))
