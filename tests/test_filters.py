import numpy
from plimg import filters


class TestFilters:
    def test_median_filter(self):
        test_arr = numpy.array([1, 1, 3, 3, 3, 3, 2, 2, 3]).reshape((3, 3))
        result_arr = filters.median(test_arr, kernel_size=1)
        expected_err = numpy.array([1, 1, 3, 3, 3, 3, 2, 2, 3]).reshape((3, 3))
        assert numpy.all(expected_err == result_arr)

    def test_median_filter_mask(self):
        test_arr = numpy.array([1, 1, 2, 3, 2, 1, 1, 2, 2]).reshape((3, 3))
        test_mask = numpy.array([True, False, False,
                                 True, True, False,
                                 True, True, False]).reshape((3, 3))
        result_arr = filters.median_mask(test_arr, 1, test_mask)
        invert_mask = numpy.invert(test_mask)

        assert numpy.all(test_arr[invert_mask] == result_arr[invert_mask])