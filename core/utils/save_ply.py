import numpy as np


def save_ply(pts: np.ndarray, rgb: np.ndarray, file_path: str = './depth.ply') -> bool:
    """
    Creates an ASCII text file containing a point cloud which is in line with the Polygon File Format (PLY).
    See https://en.wikipedia.org/wiki/PLY_(file_format) for further information.
    :param pts: numpy array [Zx3] carrying x,y,z
    :param rgb: numpy array [Zx3] carrying r,g,b
    :param file_path: file path string for file creation
    :return: True once saving succeeded
    """
    if np.max(rgb) <= 1:
        rgb *= 255
    ptsrgb = np.column_stack((pts, rgb))
    # remove invalid points
    valid_pts = ptsrgb[np.sum(np.isinf(ptsrgb) + np.isnan(ptsrgb), axis=1) == 0]

    pts_str_list = ["%010f %010f %010f %d %d %d\n" % tuple(pt) for pt in valid_pts]

    with open(file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % len(pts_str_list))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        f.writelines(pts_str_list)

    return True
