import unittest
import numpy as np

from alley_oop.geometry.lie_3d import lie_so3_to_SO3, lie_SO3_to_so3, is_SO3

#import spatialmath.base as tr

class Lie3DTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Lie3DTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_lie_conversion(self):

        arr = .25 * np.random.randn(100, 3)

        for p in arr:
            
            # convert to radians
            qs = lie_so3_to_SO3(p)

            rval = is_SO3(qs)

            self.assertTrue(rval)

            # convert to quaternions
            rs = lie_SO3_to_so3(qs)

            # assertion
            self.assertTrue(np.allclose(p, rs, atol=10e-2))

    def test_all(self):

        self.test_lie_conversion()


if __name__ == '__main__':
    unittest.main()
