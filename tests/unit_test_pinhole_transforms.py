import unittest
import numpy as np

from alley_oop.utils.pinhole_transforms import forward_project, reverse_project, projection_matrix, decompose


class PinholeTransformTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PinholeTransformTester, self).__init__(*args, **kwargs)

    def setUp(self):

        self.kmat = np.eye(3)
        self.rmat = np.eye(3)
        self.tvec = np.zeros([3, 1])

        pass

    def test_KR_decomposition(self):

        pmat = projection_matrix(self.kmat, self.rmat, self.tvec)

        kmat, rmat, tvec = decompose(pmat)

        # assertion
        concat_groundt = np.hstack([self.kmat, self.rmat, self.tvec])
        concat_results = np.hstack([kmat, rmat, tvec])
        self.assertTrue(np.allclose(concat_groundt, concat_results))

    def test_all(self):

        self.test_KR_decomposition()


if __name__ == '__main__':
    unittest.main()
