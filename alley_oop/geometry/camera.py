from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, create_img_coords_np


class PinholeCamera(object):
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def project2d(self, points3d):
        return forward_project(points3d, self.intrinsics)

    def project3d(self, points2d, depth):
        return reverse_project(points2d, self.intrinsics, depth=depth)

    def render(self, points3d, points_rgb, shape):
        img_pts, depth = forward_project(points3d, self.intrinsics, inhomogenize_opt=True)
        ipts = create_img_coords_np(shape[0], shape[1])
        # RGB
        interpolator = NearestNDInterpolator(img_pts[:2].T, points_rgb, rescale=False)
        rendered_img = interpolator(ipts[:2].T).reshape((*shape[:2], 3))
        # depth
        interpolator = NearestNDInterpolator(img_pts[:2].T, depth, rescale=False)
        rendered_depth= interpolator(ipts[:2].T).reshape(shape[:2])

        return rendered_img, rendered_depth
