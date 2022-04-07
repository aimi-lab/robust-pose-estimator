import cv2
import os
import numpy as np


def readCalibJson(settings_path, use_rectified_intrinsics=False):
    assert os.path.isfile(settings_path)
    fs = cv2.FileStorage(settings_path, cv2.FILE_STORAGE_READ)
    width = fs.getNode('Camera.width').real()
    height = fs.getNode('Camera.height').real()
    bf = fs.getNode('Camera.bf').real()
    intrinsics = np.eye(3)
    if use_rectified_intrinsics:
        intrinsics[0, 0] = fs.getNode('Camera_rect.fx').real()
        intrinsics[1, 1] = fs.getNode('Camera_rect.fy').real()
        intrinsics[0, 2] = fs.getNode('Camera_rect.cx').real()
        intrinsics[1, 2] = fs.getNode('Camera_rect.cy').real()
    else:
        intrinsics[0, 0] = fs.getNode('Camera.fx').real()
        intrinsics[1, 1] = fs.getNode('Camera.fy').real()
        intrinsics[0, 2] = fs.getNode('Camera.cx').real()
        intrinsics[1, 2] = fs.getNode('Camera.cy').real()

    return width, height, bf, intrinsics
