import os
import yaml
import numpy as np
import glob
import json
from core.utils.trajectory import save_freiburg, read_json_intuitive
import shutil

PATH_REPLACEMENT = ['/home/mhayoz/research', '/storage/workspaces/artorg_aimi/ws_00000']
if __name__ == '__main__':

    with open(os.path.join('..', 'intuitive_phantom.txt'), 'r') as f:
        sequences = f.readlines()
    for sequence in sequences:
        print(sequence)
        sequence = sequence.replace(PATH_REPLACEMENT[1], PATH_REPLACEMENT[0])
        sequence = sequence.replace('\n', '')
        if not os.path.isfile(os.path.join(sequence, 'groundtruth.txt')):
            assert os.path.isfile(os.path.join(sequence, "camera_poses.json"))
            # read poses
            poses, pose_timestamps = read_json_intuitive(os.path.join(sequence, "camera_poses.json"))
            with open(os.path.join(sequence, "IFBS_ENDOSCOPE.json"), 'r') as f:
                video_timestamps = json.load(f)
            video_timestamps = [s['timestamp'] for s in video_timestamps]
            poses_sync = []
            for tv in video_timestamps:
                # select pose that is closest to video time-stamp for synchronization
                if len(poses_sync) == 0:  # use first valid one as reference
                    ref_pose_inv = np.linalg.inv(poses[np.argmin(np.abs(pose_timestamps - tv))])
                poses_sync.append({"camera-pose": ref_pose_inv @ poses[np.argmin(np.abs(pose_timestamps - tv))], "timestamp": tv})
                #print("offset: ", np.min(np.abs(pose_timestamps - tv)))
            save_freiburg(poses_sync, sequence)
            shutil.move(os.path.join(sequence, 'trajectory.freiburg'), os.path.join(sequence, 'groundtruth.txt'))
