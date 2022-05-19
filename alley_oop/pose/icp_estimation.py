import torch
import numpy as np
import warnings
from alley_oop.geometry.lie_3d import lie_se3_to_SE3
from alley_oop.geometry.pinhole_transforms import forward_project2image, forward_project, inv_transform
from alley_oop.pose.frame_class import FrameClass
from alley_oop.fusion.surfel_map import SurfelMap
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
                 dist_thr: float=200.0/15, normal_thr: float=30, association_mode='projective'):
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
        self.src_ids = None

    @staticmethod
    def cost_fun(residuals):
        return (residuals**2).mean()

    def residual_fun(self, x, ref_pcl, target_pcl, ref_mask=None):
        T_est = lie_se3_to_SE3(x)
        self.src_ids, self.trg_ids = self.associate(ref_pcl, target_pcl, inv_transform(T_est), ref_mask)
        # compute residuals
        target_pcl_t = target_pcl.transform_cpy(T_est)
        residuals = batched_dot_product(target_pcl_t.normals.T[self.trg_ids],
                                   (ref_pcl.opts.T[self.src_ids] - target_pcl_t.opts.T[
                                       self.trg_ids]))
        return residuals

    def jacobian(self, x, ref_pcl, target_pcl, ref_mask=None):
        T_est = lie_se3_to_SE3(x)
        target_pcl_t = target_pcl.transform_cpy(T_est)
        return (target_pcl_t.normals.T[self.trg_ids].unsqueeze(1) @ self.j_3d(
            ref_pcl.opts.T[self.src_ids])).squeeze()

    def estimate_lm(self, ref_frame: FrameClass, target_pcl:SurfelMap, ref_mask: torch.tensor=None):
        """ Levenberg-Marquard estimation."""
        ref_pcl = SurfelMap(frame=ref_frame, kmat=self.intrinsics, ignore_mask=True)

        x_list, eps = lsq_lma(torch.zeros(6, device=ref_frame.depth.device, dtype=ref_frame.depth.dtype), self.residual_fun, self.jacobian,
                              args=(ref_pcl, target_pcl, ref_mask,), max_iter=self.n_iter, tol=self.Ftol, xtol=self.xtol)
        x = x_list[-1]
        cost = self.cost_fun(self.residual_fun(x, ref_pcl, target_pcl, ref_mask))
        return lie_se3_to_SE3(x), cost

    def plot(self, x, ref_pcl, target_pcl, downsample=1):
        ref_pcl = ref_pcl.transform_cpy(lie_se3_to_SE3(x))
        ref_pts = ref_pcl.opts.T.cpu().numpy()
        trg_pts = target_pcl.opts.T.cpu().numpy()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(ref_pts[::downsample, 0], ref_pts[::downsample, 1], ref_pts[::downsample, 2], c='b')
        ax.scatter(trg_pts[::downsample, 0], trg_pts[::downsample, 1], trg_pts[::downsample, 2], c='r')
        # plot point connection
        for a, b in zip(ref_pcl.opts.T[self.src_ids].cpu().numpy()[::downsample],
                        target_pcl.opts.T[self.trg_ids].cpu().numpy()[::downsample]):
            ax.plot(np.array((a[0], b[0])), np.array((a[1], b[1])), np.array((a[2], b[2])), ':', color='c', linewidth=0.5)
        plt.show()

    @staticmethod
    def j_3d(points3d): #ToDo precomputing n^T @ J_3d might speed up things
        # product of J_T*J_x for se(3)
        J = torch.zeros((len(points3d), 3, 6), dtype=points3d.dtype, device=points3d.device)
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

    def projective_association(self, ref_pcl:SurfelMap, target_pcl:SurfelMap, T_est:torch.tensor, ref_mask:torch.tensor=None):
        # update image shape
        ref_pcl = ref_pcl.transform_cpy(T_est)

        pmat_inv = inv_transform(T_est)

        # project all surfels to current image frame
        global_ipts, bidx = forward_project2image(target_pcl.opts, kmat=self.intrinsics, rmat=pmat_inv[:3, :3],
                                            tvec=pmat_inv[:3, -1][:, None], img_shape=self.img_shape)

        # find correspondence by projecting surfels to current frame
        midx = ref_pcl.get_match_indices(global_ipts[:, bidx], upscale=1)
        if ref_mask is not None:
            bidx[bidx.clone()] &= (ref_mask.view(-1)[midx]).type(torch.bool)
            midx = midx[ref_mask.view(-1)[midx]]

        # compute that rejects correspondences for a single unique one
        vidx, midx = target_pcl.filter_surfels_by_correspondence(opts=ref_pcl.opts, vidx=bidx, midx=midx,
                                                               normals=ref_pcl.normals, d_thresh=self.dist_thr,
                                                               n_thresh=self.normal_thr)

        return midx, vidx

    def check_association(self, ref_pcl: SurfelMap):
        if self.associate == self.projective_association:
            print("valid ratio:", self.trg_ids.float().mean())
        print("accuracy: ", (np.abs(ref_pcl.grid_pts[self.src_grid_ids][:, 0] - ref_pcl.opts[:, self.trg_ids][:, 0]) == 0).float().mean())

    def dist_association(self, ref_pcl: SurfelMap, target_pcl: SurfelMap, T_est: torch.tensor, ref_mask: torch.tensor=None):
        """ closest 3d euclidean distance association"""

        # update image shape
        ref_pcl = ref_pcl.transform_cpy(T_est)

        # project all surfels to current image frame
        #global_ipts, bidx = forward_project2image(target_pcl.opts, kmat=self.intrinsics, rmat=pmat_inv[:3, :3],
        #                                          tvec=pmat_inv[:3, -1][:, None], img_shape=self.img_shape)

        dists = torch.cdist(ref_pcl.opts.T.unsqueeze(0), target_pcl.opts.T.unsqueeze(0)).squeeze()
        closest_pts = torch.argmin(dists, dim=-1)
        midx = torch.arange(ref_pcl.opts.shape[1], device=ref_pcl.device)
        if ref_mask is not None:
            pass

        return midx, closest_pts
