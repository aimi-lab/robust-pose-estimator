import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrajectoryAnalyzer(object):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')

        self.pose_list = []

        if 'title' in kwargs: self.ax.set_title(kwargs['title'])
    
    def add_pose_trajectory(self, pose, label:str='', color='b', val_list=None, plot_outliers=True, linewidth=0.5, linestyle='solid'):
        """
        pose: nx3x4 nd.array
        """

        # store pose trajectory
        self.pose_list.append(pose)

        # init validation list
        val_list = np.ones(pose.shape[0]).astype('bool') if val_list is None else val_list

        # plot valid trajectory of translation vectors
        self.ax.plot(pose[val_list, 0, 3], pose[val_list, 1, 3], pose[val_list, 2, 3], linestyle=linestyle, color=color, linewidth=linewidth, markersize=3, label=label)
        self.ax.set_xlabel('x (mm)')
        self.ax.set_ylabel('y (mm)')
        self.ax.set_zlabel('z (mm)')
        # # plot trajectory outliers (if validation provided)
        # if sum(val_list) != len(val_list) and plot_outliers:
        #     self.ax.plot(pose[~val_list, 0, 3], pose[~val_list, 1, 3], pose[~val_list, 2, 3], 'x', color='b', label=str(label)+' outliers')
        #
        # # plot camera orientation as arrows
        # base_vec = np.array((0, 0, 1))
        # for i in range(pose.shape[0]):
        #     if val_list[i] or plot_outliers:
        #         rmat_es = pose[i, :3, :3]
        #         u_es, v_es, w_es = rmat_es.dot(base_vec)
        #         x_es, y_es, z_es = pose[i, :3, 3]
        #         self.ax.quiver(x_es, y_es, z_es, u_es, v_es, w_es, length=1, normalize=True, color=color)
        #
        #         # plot correspondences given alternative pose trajectory
        #         try:
        #             if len(self.pose_list) > 1: self.plot_pose_correspondence(self.pose_list[-1][i], self.pose_list[-2][i])
        #         except IndexError:
        #             break
        #
        # # plot coordinate frame of first view
        # for cc, base_vec in zip(('blue', 'red', 'green'),(np.array((1, 0, 0)),np.array((0, 1, 0)),np.array((0, 0, 1)))):
        #     rmat_es = pose[50, :3, :3]
        #     u_es, v_es, w_es = rmat_es.dot(base_vec)
        #     x_es, y_es, z_es = pose[50, :3, 3]
        #     self.ax.quiver(x_es, y_es, z_es, u_es, v_es, w_es, length=2, normalize=True, color=cc)
        #
        #     # plot correspondences given alternative pose trajectory
        #     try:
        #         if len(self.pose_list) > 1: self.plot_pose_correspondence(self.pose_list[-1][i], self.pose_list[-2][i])
        #     except IndexError:
        #         break
    def plot_pose_correspondence(self, pose_a, pose_b):

        assert pose_a.shape == pose_b.shape, 'number of pose dimensions unequal'

        x_a, y_a, z_a = pose_a[:3, 3]
        x_b, y_b, z_b = pose_b[:3, 3]

        # plot point connection
        self.ax.plot(np.array((x_b, x_a)), np.array((y_b, y_a)), np.array((z_b, z_a)), ':', color='c', linewidth=0.5)

    def get_rmse_by_idx(self, idx_a:int=-1, idx_b:int=-2, plot_opt=False):

        assert len(self.pose_list) > 1, 'found less than 2 trajectories' % len(self.pose_list)

        # if only first index is set, use its preceding trajectory
        if idx_a != -1 and idx_b == -2: idx_b = idx_a-1

        # compute RMSE
        tvec_a = self.pose_list[idx_a][:, :3, -1]
        tvec_b = self.pose_list[idx_b][:, :3, -1]
        min_len = np.min((len(tvec_a), len(tvec_b)))

        rmse = np.round(np.mean((tvec_a[:min_len]-tvec_b[:min_len])**2, axis=0)**.5, 4)

        # plot RMSE in figure
        if plot_opt: self.ax.text(x=1, y=1, z=0, s='Transl. RMSE: '+str(rmse))

        return rmse

    def write_file(self, path:str=None):

        path = Path.cwd() if path is None else Path(path)
        self.fig.savefig(path / 'pose_trajectory_plot.pdf', bbox_inches='tight')
    
    def show(self):
        self.fig.canvas.draw()
        plt.show()

    @staticmethod
    def legend(loc:str='lower right'):
        plt.legend(loc=loc)