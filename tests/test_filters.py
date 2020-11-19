import numpy
import pytest
from plimg import filters


class TestFilters:
    def test_median_filter(self):
        test_arr = numpy.array([1, 1, 3, 3, 3, 3, 2, 2, 3]).reshape((3, 3))
        result_arr = filters.median(test_arr, kernel_size=1)
        expected_err = numpy.array([1, 1, 3, 3, 3, 3, 2, 2, 3]).reshape((3, 3))
        assert numpy.all(expected_err == result_arr)

    def test_median_filter_with_mask(self):
        pass