import numpy as np
from scipy.spatial import cKDTree

def nearest_neighbour_dist(gt_pcl, predicted_pcl):
    dist_tree = cKDTree(predicted_pcl)
    dists, ids = dist_tree.query(gt_pcl, k=1)
    return dists, ids


def pcl_rms_error(gt_pcl: np.array, predicted_pcl: np.array)-> float:
    """
        RMS euclidean closest-point distances between two point-clouds
    :param gt_pcl: ground-truth 3d points nx3
    :param predicted_pcl: target 3d points mx3
    :return: rms error
    """
    return np.sqrt(np.mean(nearest_neighbour_dist(gt_pcl, predicted_pcl)[0]**2))

def pcl_absolute_error(gt_pcl: np.array, predicted_pcl: np.array)-> float:
    """
        Mean euclidean closest-point distances between two point-clouds
    :param gt_pcl: ground-truth 3d points nx3
    :param predicted_pcl: target 3d points mx3
    :return: mean absolute error
    """
    return np.mean(nearest_neighbour_dist(gt_pcl, predicted_pcl)[0])