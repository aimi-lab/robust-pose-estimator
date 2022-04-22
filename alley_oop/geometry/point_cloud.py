import torch
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, reverse_project


class PointCloud(torch.nn.Module):
    def __init__(self, pts=None, normals=None, grid_shape=None):
        super(PointCloud, self).__init__()
        self.pts = torch.nn.Parameter(pts) if pts is not None else pts
        self.normals = torch.nn.Parameter(normals) if normals is not None else normals
        self.grid_shape = grid_shape

    def estimate_normals(self):
        normals = normals_from_regular_grid(self.pts.reshape((*self.grid_shape, 3)))
        # pad normals
        pad = torch.nn.ReplicationPad2d((0,1,0,1))
        self.normals = torch.nn.Parameter(pad(normals.permute(2,1,0)).permute(1,2,0).reshape(-1,3))

    def transform(self, transform):
        assert transform.shape == (4,4)
        self.pts = self.pts @transform[:3,:3].T + transform[:3,3]
        if self.normals is not None:
            self.normals = self.normals@transform[:3,:3].T

    def transform_cpy(self, transform):
        return PointCloud(self.pts @transform[:3,:3].T + transform[:3,3], self.normals@transform[:3,:3].T, self.grid_shape).to(self.pts.device)

    def from_depth(self, depth, intrinsics, extrinsics=None):
        extrinsics = extrinsics if extrinsics is not None else torch.eye(4).to(depth.dtype).to(depth.device)
        rmat = extrinsics[:3,:3]
        tvec = extrinsics[:3, 3, None]
        img_pts = create_img_coords_t(depth.shape[0], depth.shape[1]).to(depth.dtype).to(depth.device)
        self.pts = torch.nn.Parameter(reverse_project(img_pts, intrinsics, rmat, tvec, dpth=depth).T)
        self.grid_shape = depth.shape
        self.estimate_normals()

    @property
    def grid_pts(self):
        assert self.grid_shape is not None
        return self.pts.reshape((*self.grid_shape, 3))

    @property
    def grid_normals(self):
        assert self.grid_shape is not None
        return self.normals.reshape((*self.grid_shape, 3))

