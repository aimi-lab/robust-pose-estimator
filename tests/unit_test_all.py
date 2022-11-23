from tests.unit_test_lie import Lie3DTester

test_classes = [
    Lie3DTester
    ]

for test_class in test_classes:

    # instantiate test object
    obj = test_class()
    obj.setUp()

    # switch off plots for headless tests
    obj.plot_opt = False

    obj.test_all()

    del obj
