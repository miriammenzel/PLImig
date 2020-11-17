import numpy
from plimg import histogram_toolbox


class TestHistogramToolbox:
    def test_histogram_peak_width(self):
        # Test width left
        test_arr = numpy.array([0, 0, 0.5, 0.75, 0.8, 0.85, 0.9, 1])
        width = histogram_toolbox._histogram_peak_width(test_arr,
                                                        test_arr.size - 1, -1)
        assert width == 5

        # Test width right
        test_arr = test_arr[::-1]
        width = histogram_toolbox._histogram_peak_width(test_arr, 0, 1)
        assert width == 5
