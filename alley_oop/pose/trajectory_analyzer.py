import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrajectoryAnalyzer(object):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        if '3d' in kwargs:
            self.fig = plt.figure(figsize=(7, 7))
            self.ax = self.fig.gca(projection='3d')
            self._3d = True
        else:
            self.fig, self.ax = plt.subplots(1,3,figsize=(9, 2.5))
            self._3d = False
        self.label = []

    def add_pose_trajectory(self, pose, label:str='', color='b', linewidth=0.5, linestyle='solid'):
        """
        pose: nx3x4 nd.array
        """
        self.label.append(label)
        # store pose trajectory
        if self._3d:
            # plot valid trajectory of translation vectors
            self.ax.plot(pose[:, 0, 3], pose[:, 1, 3], pose[:, 2, 3], linestyle=linestyle, color=color, linewidth=linewidth, markersize=3, label=label)
            self.ax.set_xlabel('x (mm)')
            self.ax.set_ylabel('y (mm)')
            self.ax.set_zlabel('z (mm)')

        else:
            self.ax[0].plot(pose[:, 0, 3],linestyle=linestyle, color=color, linewidth=linewidth, label=label)
            self.ax[0].set_xlabel('t (frame)')
            self.ax[0].set_ylabel('x (mm)')
            self.ax[0].grid(linestyle='dashed', linewidth=0.1)
            self.ax[1].plot(pose[:, 1, 3],linestyle=linestyle, color=color, linewidth=linewidth, label=label)
            self.ax[1].set_xlabel('t (frame)')
            self.ax[1].set_ylabel('y (mm)')
            self.ax[1].grid(linestyle='dashed', linewidth=0.1)
            self.ax[2].plot(pose[:, 2, 3],linestyle=linestyle, color=color, linewidth=linewidth, label=label)
            self.ax[2].set_xlabel('t (frame)')
            self.ax[2].set_ylabel('z (mm)')
            self.ax[2].grid(linestyle='dashed', linewidth=0.1)



    def write_file(self, path:str=None):
        plt.tight_layout()
        self.fig.savefig(path, bbox_inches='tight')
    
    def show(self):
        self.fig.canvas.draw()
        plt.tight_layout()
        plt.show()


    def legend(self):
        if self._3d:
            plt.legend()
        else:
            self.ax[2].legend()