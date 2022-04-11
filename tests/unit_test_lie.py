import unittest
import numpy as np
import torch

from alley_oop.geometry.lie_3d import lie_so3_to_SO3, lie_SO3_to_so3, is_SO3, lie_SE3_to_se3, lie_se3_to_SE3

class Lie3DTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Lie3DTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)
        torch.manual_seed(3008)

    def test_so3_conversions(self):

        arr = .25 * np.random.randn(100, 3)
        arr[0] = [0, 0, 0]
        for p in arr:
            
            # convert 3-vector to rotation matrix
            rmat = lie_so3_to_SO3(p)

            # check if rotation matrix is SO3
            rval = is_SO3(rmat)
            self.assertTrue(rval)

            # convert rotation matrix to 3-vector
            rvec = lie_SO3_to_so3(rmat)

            # assertion
            self.assertTrue(np.allclose(p, rvec, atol=10e-11))

    def test_so3_conversions_torch(self):

        arr = .25 * torch.randn(100, 3, dtype=torch.float64)

        for p in arr:
            
            # convert 3-vector to rotation matrix
            rmat = lie_so3_to_SO3(p)

            # check if rotation matrix is SO3
            rval = is_SO3(rmat)
            self.assertTrue(rval)

            # convert rotation matrix to 3-vector
            rvec = lie_SO3_to_so3(rmat)

            # assertion
            self.assertTrue(torch.allclose(p, rvec, atol=10e-11))

    def test_se3_conversions(self):

        arr = .25 * np.random.randn(100, 6)

        for p in arr:
            
            # convert 3-vector to rotation matrix
            rmat, tvec = lie_se3_to_SE3(wvec=p[:3], uvec=p[3:])

            # check if rotation matrix is SO3
            rval = is_SO3(rmat)
            self.assertTrue(rval)

            # convert rotation matrix to 3-vector
            wvec, uvec = lie_SE3_to_se3(rmat, tvec)

            # assertion
            self.assertTrue(np.allclose(p[:3], wvec, atol=10e-11))
            self.assertTrue(np.allclose(p[3:], uvec, atol=10e-11))

    def test_se3_conversions_torch(self):

        arr = .25 * torch.randn(100, 6, dtype=torch.float64)

        for p in arr:
            
            # convert 3-vector to rotation matrix
            rmat, tvec = lie_se3_to_SE3(wvec=p[:3], uvec=p[3:])

            # check if rotation matrix is SO3
            rval = is_SO3(rmat)
            self.assertTrue(rval)

            # convert rotation matrix to 3-vector
            wvec, uvec = lie_SE3_to_se3(rmat, tvec)

            # assertion
            self.assertTrue(np.allclose(p[:3], wvec, atol=10e-11))
            self.assertTrue(np.allclose(p[3:], uvec, atol=10e-11))

    def test_zero_so3(self):

        rmat = lie_so3_to_SO3(np.zeros(3))

        self.assertTrue(np.sum(rmat - np.eye(3)) == 0, 'Zero Euler angles do not yield identity matrix')

        rmat, tvec = lie_se3_to_SE3(wvec=np.zeros(3), uvec=np.zeros(3))

        self.assertTrue(np.sum((rmat - np.eye(3))**2) == 0, 'Zeros in Lie angle 3-vector do not yield identity matrix')
        self.assertTrue(np.sum(tvec**2) == 0, 'Zeros in Lie translation 3-vector are not zeros')

    def test_zero_SO3(self):

        rvec = lie_SO3_to_so3(rmat=np.eye(3))

        self.assertFalse(rvec.any(), 'Identity rotation matrix does not yield zero vector')

        rvec, uvec = lie_SE3_to_se3(rmat=np.eye(3), tvec=np.zeros(3))

        self.assertFalse(rvec.any(), 'Identity rotation matrix does not yield zero vector')
        self.assertFalse(uvec.any(), 'Zeros in translation 3-vector do not yield zeros')

    def test_all(self):

        self.test_so3_conversions()
        self.test_so3_conversions_torch()
        self.test_se3_conversions()
        self.test_se3_conversions_torch()


if __name__ == '__main__':
    unittest.main()
