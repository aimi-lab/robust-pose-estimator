import unittest
import numpy as np

from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project, compose_projection_matrix, decompose_projection_matrix, create_img_coords


class PinholeTransformTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PinholeTransformTester, self).__init__(*args, **kwargs)

    def setUp(self):

        self.kmat = np.eye(3)
        self.rmat = np.eye(3)
        self.tvec = np.zeros([3, 1])

        self.resolution = (32, 64)
        self.ipts = create_img_coords(*self.resolution)
        self.zpts = 0.1 * np.random.randn(np.multiply(*self.resolution))[np.newaxis] + 1

    def test_plane_projection(self):
        
        for plane_dist in range(1, int(1e5), 10):

            zpts = plane_dist * np.ones(np.multiply(*self.resolution))[np.newaxis]

            opts = reverse_project(self.ipts, self.kmat, self.rmat, self.tvec, disp=zpts)

            npts = forward_project(opts, self.kmat, self.rmat, self.tvec)

            self.assertTrue(np.allclose(self.ipts, npts))

    def test_KR_decomposition(self):

        pmat = compose_projection_matrix(self.kmat, self.rmat, self.tvec)

        kmat, rmat, tvec = decompose_projection_matrix(pmat)

        # assertion
        concat_groundt = np.hstack([self.kmat, self.rmat, self.tvec])
        concat_results = np.hstack([kmat, rmat, tvec])
        self.assertTrue(np.allclose(concat_groundt, concat_results))

    def test_all(self):

        self.test_KR_decomposition()


if __name__ == '__main__':
    unittest.main()
