import os
import yaml
import numpy as np
import glob
import json
from alley_oop.utils.trajectory import save_freiburg
import shutil

PATH_REPLACEMENT = ['/home/mhayoz/research', '/storage/workspaces/artorg_aimi/ws_00000']
if __name__ == '__main__':

    def load_scared_pose(fnames):

        # if kinematics data collect all files
        pose_list = []
        for i, fname in enumerate(fnames):
            with open(str(fname), 'r') as f: pose_elem = json.load(f)
            pose = np.array(pose_elem['camera-pose'])
            pose_list.append({'camera-pose': pose, 'timestamp': i})
        return pose_list

    with open(os.path.join('..', 'scared.txt'), 'r') as f:
        sequences = f.readlines()
    for sequence in sequences:
        print(sequence)
        sequence = sequence.replace(PATH_REPLACEMENT[1], PATH_REPLACEMENT[0])
        sequence = sequence.replace('\n', '')
        if not os.path.isfile(os.path.join(sequence, 'groundtruth.txt')):
            gt_files = sorted(glob.glob(os.path.join(sequence, 'data', 'frame_data', '*.json')))
            assert len(gt_files) > 0
            pose_list = load_scared_pose(gt_files)
            save_freiburg(pose_list, sequence)
            shutil.move(os.path.join(sequence, 'trajectory.freiburg'), os.path.join(sequence, 'groundtruth.txt'))
