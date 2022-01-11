import numpy as np


def forward_project(obj_pts, cam_mtx, pos_mtx=None):

    pos_mtx = np.hstack([np.eye(3), np.zeros([3, 1])]) if pos_mtx is None else pos_mtx

    # pinhole projection
    img_pts = pos_mtx @ obj_pts

    # inhomogenization
    img_pts = img_pts / img_pts[-1]

    # intrinsic scaling
    img_pix = cam_mtx @ img_pts

    return img_pix


def reverse_project(img_pix, cam_mtx, pos_mtx=None, disp=None, base=float(1)):

    pos_mtx = np.hstack([np.eye(3), np.zeros([3, 1])]) if pos_mtx is None else pos_mtx

    # intrinsic scaling
    img_pts = np.linalg.inv(cam_mtx) @ img_pix

    # pinhole projection
    obj_pts = np.linalg.pinv(pos_mtx) @ img_pts

    # depth assignment
    if disp is not None:
        obj_pts = base * np.diag([cam_mtx[0][0], cam_mtx[1][1], np.mean([cam_mtx[0][0], cam_mtx[1][1]])]) @ obj_pts[:3] / disp

    return obj_pts