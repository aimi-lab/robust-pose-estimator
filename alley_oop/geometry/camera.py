from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project


class PinholeCamera(object):
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def project2d(self, points3d):
        return forward_project(points3d, self.intrinsics)

    def project3d(self, points2d, depth):
        return reverse_project(points2d, self.intrinsics, depth=depth)
