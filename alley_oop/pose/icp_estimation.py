import torch
import numpy as np
from alley_oop.geometry.lie_3d import lie_se3_to_SE3
from alley_oop.geometry.pinhole_transforms import forward_project
from alley_oop.geometry.point_cloud import PointCloud
from alley_oop.utils.pytorch import batched_dot_product
from typing import Tuple
import matplotlib.pyplot as plt
from torchimize.functions import lsq_lma


class ICPEstimator(torch.nn.Module):
    """ this is an implementation of the geometric alignment in Elastic Fusion
        references: Newcombe et al, Kinect Fusion Chapter 3.5 (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6162880)
                    Henry et al, Patch Volumes: Segmentation-based Consistent Mapping with RGB-D Cameras (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6599102)
    It estimates the camera rotation and translation between a scene and a current depth map
"""

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, n_iter: int=20, Ftol: float=0.001, xtol: float=1e-8,
                 dist_thr: float=200.0/15, normal_thr: float=0.94, association_mode='projective'):
        """

        :param img_shape: height and width of images to process
        :param intrinsics: camera intrinsics
        :param n_iter: max number of iterations of the optimizer
        :param Ftol: cost function threshold
        :param xtol: variable change threshold
        :param dist_thr: euclidean distance threshold to accept/reject point correspondences
        :param normal_thr: angular difference threshold of normals to accept/reject point correspondences
        :param association_mode: projective or euclidean correspondence in ICP
        """
        super(ICPEstimator, self).__init__()
        assert len(img_shape) == 2
        assert intrinsics.shape == (3,3)
        assert association_mode in ['projective', 'euclidean']
        self.associate = self.dist_association if association_mode == 'euclidean' else self.projective_association
        self.img_shape = img_shape
        self.n_iter = n_iter
        self.Ftol = Ftol
        self.xtol = xtol
        self.dist_thr = dist_thr
        self.normal_thr = normal_thr
        self.intrinsics = torch.nn.Parameter(intrinsics)
        self.d = torch.nn.Parameter(torch.empty(0))  # dummy device store
        self.trg_ids = None
        self.src_grid_ids = None

    @staticmethod
    def cost_fun(residuals):
        return (residuals**2).mean()

    def residual_fun(self, x, ref_pcl, target_pcl, mask=None):
        xt = torch.tensor(x).double().to(ref_pcl.pts.device) if not torch.is_tensor(x) else x
        T_est = lie_se3_to_SE3(xt[:3], xt[3:], homogenous=True)
        target_pcl_current_c = target_pcl.transform_cpy(torch.linalg.inv(T_est))  # transform into current frame
        self.src_grid_ids, self.trg_ids = self.associate(ref_pcl, target_pcl_current_c, mask)
        # compute residuals
        ref_pcl_world_c = ref_pcl.transform_cpy(T_est)
        return batched_dot_product(target_pcl.normals[self.trg_ids],
                                   (ref_pcl_world_c.grid_pts[self.src_grid_ids] - target_pcl.pts[
                                       self.trg_ids]))

    def jacobian(self, x, ref_pcl, target_pcl, mask=None):
        xt = torch.tensor(x).double() if not torch.is_tensor(x) else x
        T_est = lie_se3_to_SE3(xt[:3], xt[3:], homogenous=True)
        ref_pcl_world_c = ref_pcl.transform_cpy(T_est)
        return (target_pcl.normals[self.trg_ids].unsqueeze(1) @ self.j_3d(
            ref_pcl_world_c.grid_pts[self.src_grid_ids])).squeeze()

    def estimate_lm(self, ref_depth: torch.Tensor, target_pcl:PointCloud, mask: torch.tensor=None):
        """ Levenberg-Marquard estimation."""
        ref_pcl = PointCloud()
        ref_pcl.from_depth(ref_depth, self.intrinsics)

        x_list, eps = lsq_lma(torch.zeros(6).to(ref_depth.device).to(ref_depth.dtype), self.residual_fun, self.jacobian,
                              args=(ref_pcl, target_pcl,mask,), max_iter=self.n_iter, tol=self.Ftol, xtol=self.xtol)
        x = x_list[-1]
        cost = self.cost_fun(self.residual_fun(x, ref_pcl, target_pcl, mask))
        return lie_se3_to_SE3(x[:3], x[3:], homogenous=True), cost

    def plot(self, x, ref_pcl, target_pcl, downsample=1):
        ref_pcl = ref_pcl.transform_cpy(lie_se3_to_SE3(x[:3], x[3:], homogenous=True))
        ref_pts = ref_pcl.pts.cpu().numpy()
        trg_pts = target_pcl.pts.cpu().numpy()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(ref_pts[::downsample, 0], ref_pts[::downsample, 1], ref_pts[::downsample, 2], c='b')
        ax.scatter(trg_pts[::downsample, 0], trg_pts[::downsample, 1], trg_pts[::downsample, 2], c='r')
        # plot point connection
        for a, b in zip(ref_pcl.grid_pts[self.src_grid_ids].cpu().numpy()[::downsample],
                        target_pcl.pts[self.trg_ids].cpu().numpy()[::downsample]):
            ax.plot(np.array((a[0], b[0])), np.array((a[1], b[1])), np.array((a[2], b[2])), ':', color='c', linewidth=0.5)
        plt.show()

    def plot_correspondence(self, x, ref_img, ref_depth, target_img):
        from alley_oop.interpol.synth_view import synth_view
        R, t = lie_se3_to_SE3(x[:3].cpu(), x[3:].cpu())
        warped_img = synth_view(ref_img.unsqueeze(0).unsqueeze(0).cpu().float(), ref_depth.unsqueeze(0).float().cpu(),
                                R.float(),
                                t.unsqueeze(1).float(), self.intrinsics.float().cpu()).squeeze()
        ref_img1 = ref_img.clone()
        ref_img1[self.src_grid_ids[0][0], self.src_grid_ids[1][0]] = 1
        trg_ids = torch.where(self.trg_ids.reshape(self.img_shape))
        target_img1 = target_img.clone()
        target_img1[trg_ids[0][0], trg_ids[1][0]] = 1
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(ref_img1.cpu())
        ax[0].set_title('reference')
        ax[1].imshow(target_img1.cpu())
        ax[1].set_title('target')
        ax[2].imshow(warped_img.cpu())
        ax[2].set_title('estimated')
        for a in ax:
            a.axis('off')
        plt.show()


    @staticmethod
    def j_3d(points3d): #ToDo precomputing n^T @ J_3d might speed up things
        # product of J_T*J_x for se(3)
        J = torch.zeros((len(points3d), 3, 6), dtype=points3d.dtype).to(points3d.device)
        J[:, 0, 1] = points3d[:, 2]
        J[:, 0, 2] = -points3d[:, 1]
        J[:, 0, 3] = 1
        J[:, 1, 0] = -points3d[:, 2]
        J[:, 1, 2] = points3d[:, 0]
        J[:, 1, 4] = 1
        J[:, 2, 0] = points3d[:, 1]
        J[:, 2, 1] = -points3d[:, 0]
        J[:, 2, 5] = 1
        return J

    def projective_association(self, src_pcl:PointCloud, target_pcl:PointCloud, mask:torch.tensor=None):
        """ perform projectiv data associcaton"""
        extrinsics = torch.eye(4).to(target_pcl.pts.dtype).to(target_pcl.pts.device)
        rmat = extrinsics[:3, :3]
        tvec = extrinsics[:3, 3, None]
        pts_h = torch.vstack([target_pcl.pts.T, torch.ones(target_pcl.pts.shape[0],
                                                           device=target_pcl.pts.device, dtype=target_pcl.pts.dtype)])
        points_2d = forward_project(pts_h, self.intrinsics, rmat=rmat, tvec=tvec).T

        # filter points that are not in the image
        valid = (points_2d[:, 1] < self.img_shape[0]) & (points_2d[:, 0] < self.img_shape[1]) & (points_2d[:, 1] > 0) & (points_2d[:, 0] > 0)
        if mask is not None:
            valid[valid.clone()] &= (mask[points_2d[valid][:,1].long(),points_2d[valid][:,0].long()]).type(torch.bool)

        # filter points that are too far in 3d space, or that have a large angle between the normals
        ids = points_2d.long()[valid][:, 1], points_2d.long()[valid][:, 0]
        valid_dist = torch.norm(src_pcl.grid_pts[ids] - target_pcl.pts[valid], p=2, dim=-1) < self.dist_thr
        valid_normal = batched_dot_product(src_pcl.grid_normals[ids], target_pcl.normals[valid]) > self.normal_thr
        valid[valid.clone()] &= valid_dist & valid_normal
        ids = points_2d.long()[valid][:, 1], points_2d.long()[valid][:, 0]
        return ids, valid

    def check_association(self, src_pcl: PointCloud):
        if self.associate == self.projective_association:
            print("valid ratio:", self.trg_ids.float().mean())
        print("accuracy: ", (np.abs(src_pcl.grid_pts[self.src_grid_ids][:,0]- src_pcl.pts[self.trg_ids][:,0]) == 0).float().mean())

    @staticmethod
    def dist_association(src_pcl: PointCloud, target_pcl: PointCloud, mask: torch.tensor=None):
        """ closest 3d euclidean distance association"""

        from scipy.spatial import KDTree
        #ToDo add masking, support direct torch
        tree = KDTree(target_pcl.pts.numpy())
        closest_pts = tree.query(src_pcl.pts.numpy())[1]
        ids = torch.meshgrid(torch.arange(src_pcl.grid_shape[0]), torch.arange(src_pcl.grid_shape[1]))
        ids = (ids[0].reshape(-1), ids[1].reshape(-1))
        valid = torch.tensor(closest_pts)
        return ids, valid
