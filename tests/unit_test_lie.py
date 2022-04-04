import unittest
import numpy as np

from alley_oop.geometry.lie_3d import lie_algebra2group_rot, lie_group2algebra_rot


class Lie3DTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Lie3DTester, self).__init__(*args, **kwargs)

    def setUp(self):

        pass

    def test_lie_conversion(self):

        arr = .1*np.random.randn(100, 3)

        for p in arr:
            
            # convert to radians
            qs = lie_algebra2group_rot(p)

            # convert to quaternions
            rs = lie_group2algebra_rot(qs)

            # assertion
            self.assertTrue(np.allclose(p, rs, atol=10e-3))

    def test_all(self):

        self.test_lie_conversion()


if __name__ == '__main__':
    unittest.main()
