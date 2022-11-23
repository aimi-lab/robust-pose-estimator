from tests.unit_test_normals import NormalsTester
from tests.unit_test_photo_loss import PhotoLossTest
from tests.unit_test_quaternions import QuaternionConversionTester
from tests.unit_test_lie import Lie3DTester
from tests.unit_test_gauss_pyramid import GaussPyramidTester
from tests.unit_test_grayscale import GrayscaleTester
from tests.unit_test_surfel_map import SurfelMapTest
from tests.unit_test_pose_head import PoseHeadTester

test_classes = [
    NormalsTester, PhotoLossTest, Lie3DTester, QuaternionConversionTester,
    GaussPyramidTester, GrayscaleTester, SurfelMapTest,
    PoseHeadTester
    ]

for test_class in test_classes:

    # instantiate test object
    obj = test_class()
    obj.setUp()

    # switch off plots for headless tests
    obj.plot_opt = False

    obj.test_all()

    del obj
