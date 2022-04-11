import torch
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, reverse_project


class PointCloud(object):
    def __init__(self, pts=None, normals=None):
        self.pts = pts
        self.normals = normals

    def estimate_normals(self, grid_shape):
        normals = normals_from_regular_grid(self.pts.reshape((*grid_shape, 3)))
        # pad normals
        self.normals =torch.nn.functional.pad(normals, (0,0,0,1,0,1))

    def transform(self, transform):
        assert transform.shape == (4,4)
        self.pts = self.pts @transform[:3,:3].T + transform[:3,3]
        if self.normals is not None:
            self.normals = self.normals@transform[:3,:3].T

    def transform_cpy(self, transform):
        return PointCloud(self.pts @transform[:3,:3].T + transform[:3,3], self.normals@transform[:3,:3].T)

    def from_depth(self, depth, intrinsics, extrinsics=None):
        rmat = extrinsics[:3,:3] if extrinsics is not None else None
        tvec = extrinsics[:3, 3] if extrinsics is not None else None
        img_pts = create_img_coords_t(depth.shape[0], depth.shape[1])
        self.pts = reverse_project(img_pts, intrinsics, rmat, tvec, depth=depth).T
        self.estimate_normals(depth.shape)
