import wandb
from scipy.spatial.transform import Rotation as R
import numpy as np


class OptimizationRecordings():
    def __init__(self):
        self.trajectory = []
        self.surfels_total = []
        self.surfels_stable = []
        self.gt_trajectory = None

    def __del__(self):
        wandb.finish()

    def __call__(self, scene, pose):
        if scene is not None:
            self.surfels_total.append(scene.opts.shape[1])
            self.surfels_stable.append((scene.conf >= 1.0).sum().item())
        else:
            self.surfels_total.append(0)
            self.surfels_stable.append(0)
        self.trajectory.append(pose.cpu().numpy())


    def log(self, step):
        log_dict = {'frame': step,
                   'surfels/total': self.surfels_total[-1],
                   'surfels/stable': self.surfels_stable[-1]}

        if self.gt_trajectory is not None:
            if len(self.gt_trajectory) > step:
                tr_err = self.gt_trajectory[step][:3,3] - self.trajectory[-1][:3,3]
                rot_err = (self.gt_trajectory[step][:3, :3].T @ self.trajectory[-1][:3, :3])
                rot_err_deg = np.linalg.norm(R.from_matrix(rot_err).as_rotvec(degrees=True), ord=2)
                log_dict.update({'error/x': tr_err[0],
                                 'error/y': tr_err[1],
                                 'error/z': tr_err[2],
                                 'error/rot': rot_err_deg,
                                 'error/x_pred': self.trajectory[-1][0, 3],
                                 'error/y_pred': self.trajectory[-1][1, 3],
                                 'error/z_pred': self.trajectory[-1][2, 3],
                                 'error/x_gt': self.gt_trajectory[step][0,3],
                                 'error/y_gt': self.gt_trajectory[step][1,3],
                                 'error/z_gt':self.gt_trajectory[step][2,3]})
        wandb.log(log_dict, step=step)

    def set_gt(self, gt_trajectory):
        self.gt_trajectory = gt_trajectory

