import cv2
import numpy as np

def kpts2npy(cv2kpts):
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in cv2kpts])
    return kpts

def npy2kpts(np_pts):
    cv2kpts = [cv2.KeyPoint(pt[0],pt[1]) for pt in np_pts]
    return cv2kpts