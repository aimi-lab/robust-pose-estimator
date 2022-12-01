from tests.unit_test_pinhole_transforms import PinholeTransformTester

test_classes = [
    PinholeTransformTester
    ]

for test_class in test_classes:

    # instantiate test object
    obj = test_class()
    obj.setUp()

    # switch off plots for headless tests
    obj.plot_opt = False

    obj.test_all()

    del obj
