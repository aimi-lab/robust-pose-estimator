import unittest
import numpy as np

from alley_oop.geometry.quaternions import euler2quat, quat2euler


class QuaternionConverionTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(QuaternionConverionTester, self).__init__(*args, **kwargs)

    def setUp(self):

        pass

    def test_quaternion_conversion(self):

        arr = .1*np.random.randn(100, 3)+1

        for p in arr:
            
            # convert to radians
            qs = euler2quat(*p)

            # convert to quaternions
            rs = quat2euler(qs)

            # assertion
            self.assertTrue(np.allclose(p, rs))

    def test_all(self):

        self.test_quaternion_conversion()


if __name__ == '__main__':
    unittest.main()
