import torch
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, reverse_project
from alley_oop.pose.frame_class import FrameClass


class PointCloud(torch.nn.Module):
    def __init__(self, pts=None, normals=None, grid_shape=None, colors=None):
        super(PointCloud, self).__init__()
        self.pts = torch.nn.Parameter(pts) if pts is not None else pts
        self.normals = torch.nn.Parameter(normals) if normals is not None else normals
        self.colors = torch.nn.Parameter(colors) if colors is not None else colors
        self.grid_shape = grid_shape

    def estimate_normals(self):
        normals = normals_from_regular_grid(self.pts.reshape((*self.grid_shape, 3)))
        # pad normals
        pad = torch.nn.ReplicationPad2d((0,1,0,1))
        self.normals = torch.nn.Parameter(pad(normals.permute(2,1,0)).permute(1,2,0).reshape(-1,3))

    def transform(self, transform):
        assert transform.shape == (4,4)
        self.pts.data = self.pts @transform[:3,:3].T + transform[:3,3]
        if self.normals is not None:
            self.normals.data = self.normals@transform[:3,:3].T

    def transform_cpy(self, transform):
        return PointCloud(self.pts @transform[:3,:3].T + transform[:3,3], self.normals@transform[:3,:3].T, self.grid_shape, self.colors).to(self.pts.device)

    def from_depth(self, depth, intrinsics, extrinsics=None, normals=None):
        extrinsics = extrinsics if extrinsics is not None else torch.eye(4).to(depth.dtype).to(depth.device)
        rmat = extrinsics[:3,:3]
        tvec = extrinsics[:3, 3, None]
        img_pts = create_img_coords_t(depth.shape[-2], depth.shape[-1]).to(depth.dtype).to(depth.device)
        self.pts = torch.nn.Parameter(reverse_project(img_pts, intrinsics, rmat, tvec, dpth=depth).T)
        self.grid_shape = depth.shape[-2:]
        if normals is None:
            self.estimate_normals()
        else:
            self.normals = torch.nn.Parameter(normals.view(-1,3))

    def set_colors(self, colors):
        self.colors = torch.nn.Parameter(colors.squeeze().permute(1,2,0).reshape(-1, 3))

    @property
    def grid_pts(self):
        assert self.grid_shape is not None
        return self.pts.view((*self.grid_shape, 3))

    @property
    def grid_normals(self):
        assert self.grid_shape is not None
        return self.normals.view((*self.grid_shape, 3))

    def render(self, intrinsics):
        from alley_oop.geometry.pinhole_transforms import forward_project
        extrinsics = torch.eye(4).to(self.pts.dtype).to(self.pts.device)
        rmat = extrinsics[:3, :3]
        tvec = extrinsics[:3, 3, None]
        pts_h = torch.vstack([self.pts.T, torch.ones(self.pts.shape[0],
                                                           device=self.pts.device, dtype=self.pts.dtype)])
        points_2d = forward_project(pts_h, intrinsics, rmat=rmat, tvec=tvec).T

        # filter points that are not in the image
        valid = (points_2d[:, 1] < self.grid_shape[0]) & (points_2d[:, 0] < self.grid_shape[1]) & (
                    points_2d[:, 1] > 0) & (points_2d[:, 0] > 0)
        points_2d = points_2d[valid][...,:2]
        colors = self.colors[valid]

        import numpy as np
        x_coords = np.arange(0, self.grid_shape[1]) + 0.5
        y_coords = np.arange(0, self.grid_shape[0]) + 0.5
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
        interp = NDInterpolator(points_2d, colors, dist_thr=10, default_value=0)
        interp.fit(ipts, self.grid_shape)
        img = torch.stack([interp.predict(colors[:, i]) for i in range(3)])
        return FrameClass(img[None,:], interp.predict(self.pts[:,2])[None,None,:], intrinsics=intrinsics).to(intrinsics.device)


from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.interpolate import NearestNDInterpolator
import torch


class NDInterpolator(NearestNDInterpolator):
    def __init__(self, x, y, rescale=False, tree_options=None, dist_thr=10, default_value=0.0):
        if y.ndim > 1:
            y_cpu = y[:,0].cpu().numpy()
        else:
            y_cpu = y.cpu().numpy()
        NearestNDInterpolator.__init__(self, x.cpu().numpy(), y_cpu, rescale=rescale, tree_options=tree_options)
        self.dist_thr = dist_thr
        self.default_value = default_value

    def fit(self, in_tensor, shape):
        """
        Evaluate interpolator at given points.
        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.
        """

        xi = _ndim_coords_from_arrays(in_tensor, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)
        dist, i = self.tree.query(xi)
        self.i = i
        self.shape = shape
        self.dist_mask = dist > self.dist_thr

    def predict(self, y):
        interp = y[self.i]
        interp[self.dist_mask] = self.default_value
        return interp.view(self.shape[:y.ndim+1])
