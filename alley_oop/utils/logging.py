import wandb
from scipy.spatial.transform import Rotation as R
import numpy as np


class OptimizationRecordings():
    def __init__(self, pyramid_levels):
        self.costs_combined = [[] for i in range(pyramid_levels)]
        self.costs_rgb = [[] for i in range(pyramid_levels)]
        self.costs_icp = [[] for i in range(pyramid_levels)]
        self.trajectory = []
        self.surfels_total = []
        self.surfels_stable = []
        self.pyramid_levels = pyramid_levels
        self.gt_trajectory = None

    def __call__(self, scene, estimator, pose):
        self.surfels_total.append(scene.opts.shape[1])
        self.surfels_stable.append((scene.conf >= 1.0).sum().item())
        self.trajectory.append(pose.cpu().numpy())
        for i in range(self.pyramid_levels):
            self.costs_combined[i].append(estimator.cost[i][0])
            self.costs_icp[i].append(estimator.cost[i][1])
            self.costs_rgb[i].append(estimator.cost[i][2])

    def log(self, step):
        log_dict = {'frame': step,
                   'surfels/total': self.surfels_total[-1],
                   'surfels/stable': self.surfels_stable[-1]}
        for i in range(self.pyramid_levels):
            log_dict.update({f'pyr{i}/cost': self.costs_combined[i][-1],
                       f'pyr{i}/cost_rgb': self.costs_icp[i][-1],
                       f'pyr{i}/cost_icp': self.costs_rgb[i][-1]})
        if self.gt_trajectory is not None:
            if len(self.gt_trajectory) > step:
                tr_err = self.gt_trajectory[step][:3,3] - self.trajectory[-1][:3,3]
                rot_err = (self.gt_trajectory[step][:3, :3].T @ self.trajectory[-1][:3, :3])
                rot_err_deg = np.linalg.norm(R.from_matrix(rot_err).as_rotvec(degrees=True), ord=2)
                log_dict.update({'error/x': tr_err[0],
                                 'error/y': tr_err[1],
                                 'error/z': tr_err[2],
                                 'error/rot': rot_err_deg})
        wandb.log(log_dict, step=step)

    def set_gt(self, gt_trajectory):
        self.gt_trajectory = gt_trajectory

    def plot(self, show=False):
        if not show:
            import matplotlib as mpl
            mpl.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(self.pyramid_levels+1,1)

        for i in range(self.pyramid_levels):
            ax[i].set_title(f'Optimization Cost at Pyramid Lv {i}')
            ax[i].plot([j.item() for j in self.costs_combined[i]])
            ax[i].plot([j.item() for j in self.costs_icp[i]])
            ax[i].plot([j.item() for j in self.costs_rgb[i]])
            ax[i].set_xlim([0, 1.1*max([j.item() for j in self.costs_combined[i]])])
            ax[i].grid()
            ax[0].legend(['combined', 'icp', 'rgb'])
        ax[self.pyramid_levels].plot(self.surfels_stable)
        ax[self.pyramid_levels].plot(self.surfels_total)
        ax[self.pyramid_levels].legend(['stable', 'unstable'])
        ax[self.pyramid_levels].set_title(f'Number of Surfels')
        ax[self.pyramid_levels].grid()
        ax[self.pyramid_levels].set_xlabel('time [frames]')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
