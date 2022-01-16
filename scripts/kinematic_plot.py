from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def kinametic_plot(pose_es, pose_gt=None, val_list=None, plot_outliers=True, save_opt=False, name=None):
    
    val_list = np.ones(pose_es.shape[0]).astype('bool') if val_list is None else val_list
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(name)

    # plot keyframe point
    ax.plot([pose_gt[0, 0, 3]], [pose_gt[0, 1, 3]], [pose_gt[0, 2, 3]], marker='s', color='orange', markersize=6, linestyle='none', label='keyframe')

    # plot valid trajectory of translation vectors
    ax.plot(pose_es[val_list, 0, 3], pose_es[val_list, 1, 3], pose_es[val_list, 2, 3], '-s', color='b', linewidth=0.5, markersize=3, label='pose estimation')
    ax.plot(pose_gt[val_list, 0, 3], pose_gt[val_list, 1, 3], pose_gt[val_list, 2, 3], '-o', color='g', linewidth=0.5, markersize=3, label='kinematics (ground-truth)')

    # plot trajectory outliers (if validation provided)
    if sum(val_list) != len(val_list) and plot_outliers:
        ax.plot(pose_es[~val_list, 0, 3], pose_es[~val_list, 1, 3], pose_es[~val_list, 2, 3], 'x', color='b', label='estimation outliers')
        ax.plot(pose_gt[~val_list, 0, 3], pose_gt[~val_list, 1, 3], pose_gt[~val_list, 2, 3], 'x', color='g' if plot_outliers else 'r', label='kinematics outliers')

    plt.legend()

    base_vec = np.array((0, 0, -1))
    for i in range(pose_gt.shape[0]):
        if val_list[i] or plot_outliers:

            # plot camera orientation arrows
            rmat_es = pose_es[i, :3, :3]
            u_es, v_es, w_es = rmat_es.dot(base_vec)
            x_es, y_es, z_es = pose_es[i, :3, 3]
            ax.quiver(x_es, y_es, z_es, u_es, v_es, w_es, length=2, normalize=True, color='b')
            rmat_gt = pose_gt[i, :3, :3]
            u_gt, v_gt, w_gt = rmat_gt.dot(base_vec)
            x_gt, y_gt, z_gt = pose_gt[i, :3, 3]
            ax.quiver(x_gt, y_gt, z_gt, u_gt, v_gt, w_gt, length=2, normalize=True, color='g')

            # plot point connection
            ax.plot(np.array((x_gt, x_es)), np.array((y_gt, y_es)), np.array((z_gt, z_es)), ':', color='c', linewidth=0.5)

    if save_opt: plt.savefig(str(Path().cwd() / 'kinematics_plot.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    data_path = Path.cwd() / 'tests' / 'test_data'
    name_list = sorted((data_path).rglob('pose*.npz'))

    for fname in name_list:
        npz_obj = np.load(fname, allow_pickle=True)
        pose_gt = np.array([pmat[:3, :] for pmat in npz_obj[npz_obj.files[0]]])[1::2, ...]
        pose_es = np.array([pmat[:3, :] for pmat in npz_obj[npz_obj.files[1]]])[1::2, ...]

        kinametic_plot(pose_es, pose_gt, name=fname.name)
