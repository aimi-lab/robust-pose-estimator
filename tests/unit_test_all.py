from tests.unit_test_pose import PoseTester
from tests.unit_test_normals import NormalsTester
from tests.unit_test_pinhole_transforms import PinholeTransformTester
from tests.unit_test_photo_loss import PhotoLossTest
from tests.unit_test_quaternions import QuaternionConversionTester
from tests.unit_test_rot_estimation import RotEstimatorTester
from tests.unit_test_lie import Lie3DTester

test_classes = [
    PoseTester, NormalsTester, PinholeTransformTester, PhotoLossTest, Lie3DTester, QuaternionConversionTester,
    RotEstimatorTester,
                ]

for test_class in test_classes:

    # instantiate test object
    obj = test_class()
    obj.setUp()

    # switch off plots for headless
    obj.plt_opt = False

    obj.test_all()

    del obj
