import unittest
import numpy as np
import torch

from alley_oop.geometry.lie_3d import lie_so3_to_SO3, lie_SO3_to_so3, is_SO3

class Lie3DTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Lie3DTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)
        torch.manual_seed(3008)

    def test_lie_conversion(self):

        arr = .25 * np.random.randn(100, 3)
        arr[0] = [0, 0, 0]
        for p in arr:
            
            # convert 3-vector to rotation matrix
            qs = lie_so3_to_SO3(p)

            # check if rotation matrix is SO3
            rval = is_SO3(qs)
            self.assertTrue(rval)

            # convert rotation matrix to 3-vector
            rs = lie_SO3_to_so3(qs)

            # assertion
            self.assertTrue(np.allclose(p, rs, atol=10e-11))

    def test_lie_conversion_torch(self):

        arr = .25 * torch.randn(100, 3, dtype=torch.float64)

        for p in arr:
            
            # convert 3-vector to rotation matrix
            qs = lie_so3_to_SO3(p)

            # check if rotation matrix is SO3
            rval = is_SO3(qs)
            self.assertTrue(rval)

            # convert rotation matrix to 3-vector
            rs = lie_SO3_to_so3(qs)

            # assertion
            self.assertTrue(torch.allclose(p, rs, atol=10e-11))

    def test_all(self):

        self.test_lie_conversion_torch()
        self.test_lie_conversion()


if __name__ == '__main__':
    unittest.main()
