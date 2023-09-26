import wandb
from scipy.spatial.transform import Rotation as R
import numpy as np


class InferenceLogger:
    def __init__(self):
        self.gt_trajectory = None

    def __del__(self):
        wandb.finish()

    def __call__(self, scene, pose, step):
        if scene is not None:
            surfels_total = scene.opts.shape[1]
            surfels_stable = (scene.conf >= 1.0).sum().item()
        else:
            surfels_total = 0
            surfels_stable = 0

        log_dict = {'frame': step,
                    'surfels/total': surfels_total,
                    'surfels/stable': surfels_stable}
        pose = pose.matrix().squeeze().cpu().numpy()
        if self.gt_trajectory is not None:
            if len(self.gt_trajectory) > step:
                tr_err = self.gt_trajectory[step][:3, 3] - pose[:3, 3]
                rot_err = (self.gt_trajectory[step][:3, :3].T @ pose[:3, :3])
                rot_err_deg = np.linalg.norm(R.from_matrix(rot_err).as_rotvec(degrees=True), ord=2)
                euler_pred = R.from_matrix(pose[:3, :3]).as_euler('zxy',degrees=True)
                euler_gt = R.from_matrix(self.gt_trajectory[step][:3, :3]).as_euler('zxy',degrees=True)
                log_dict.update({'error/x': tr_err[0],
                                 'error/y': tr_err[1],
                                 'error/z': tr_err[2],
                                 'error/rot': rot_err_deg,
                                 'error/x_pred': pose[0, 3],
                                 'error/y_pred': pose[1, 3],
                                 'error/z_pred': pose[2, 3],
                                 'error/alpha_pred': euler_pred[0],
                                 'error/beta_pred': euler_pred[1],
                                 'error/gamma_pred': euler_pred[2],
                                 'error/x_gt': self.gt_trajectory[step][0, 3],
                                 'error/y_gt': self.gt_trajectory[step][1, 3],
                                 'error/z_gt': self.gt_trajectory[step][2, 3],
                                 'error/alpha_gt': euler_gt[0],
                                 'error/beta_gt': euler_gt[1],
                                 'error/gamma_gt': euler_gt[2],
                                 })
        wandb.log(log_dict, step=step)

    def set_gt(self, gt_trajectory):
        self.gt_trajectory = gt_trajectory.matrix()


class TrainLogger:
    def __init__(self, model, config, project_name, log):
        self.model = model
        self.total_steps = 0
        self.running_loss = {'train': {}, 'val': {}}
        self.log = log
        if log:
            wandb.init(project=project_name, config=config)
        self.header = False

    def _print_header(self):
        metrics_data = [k for k in sorted(self.running_loss['train'].keys())]
        metrics_str = ("{:<15}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(metrics_str)

    def _print_training_status(self, mode):
        if not self.header:
            self.header = True
            self._print_header()
        metrics_data = [self.running_loss[mode][k] for k in sorted(self.running_loss[mode].keys())]
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(metrics_str)

        for k in self.running_loss[mode]:
            self.running_loss[mode][k] = 0.0

    def push(self, metrics, freq, mode='train'):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss[mode]:
                self.running_loss[mode][key] = 0.0

            self.running_loss[mode][key] += metrics[key] / freq

    def write_dict(self, results):
        wandb.log(results)

    def flush(self, mode='train'):
        if self.log:
            self.write_dict(self.running_loss[mode])
        self._print_training_status(mode)
        self.running_loss[mode] = {}

    def close(self):
        wandb.finish()

    def save_model(self, path):
        if self.log:
            wandb.save(path)

    def log_plot(self, fig):
        if self.log:
            wandb.log({"optical flow": fig})