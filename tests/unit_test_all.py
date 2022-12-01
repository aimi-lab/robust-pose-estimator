from tests.unit_test_pinhole_transforms import PinholeTransformTester
from tests.unit_test_pose_head import PoseHeadTester

test_classes = [
    PinholeTransformTester, PoseHeadTester
    ]

for test_class in test_classes:

    # instantiate test object
    obj = test_class()
    obj.setUp()

    # switch off plots for headless tests
    obj.plot_opt = False

    obj.test_all()

    del obj
