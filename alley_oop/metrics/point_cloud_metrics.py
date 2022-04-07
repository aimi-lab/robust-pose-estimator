import numpy as np
from scipy.spatial import cKDTree

def nearest_neighbour_dist(gt_pcl, predicted_pcl):
    dist_tree = cKDTree(gt_pcl)
    dists = dist_tree.query(predicted_pcl, k=1)[0]
    return dists


def pcl_rmse(gt_pcl, predicted_pcl):
    return np.sqrt(np.mean(nearest_neighbour_dist(gt_pcl, predicted_pcl)**2))

def pcl_ae(gt_pcl, predicted_pcl):
    return np.mean(nearest_neighbour_dist(gt_pcl, predicted_pcl))
