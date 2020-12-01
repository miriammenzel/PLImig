//
// Created by jreuter on 27.11.20.
//
#include "gtest/gtest.h"
#include <memory>
#include <opencv2/core.hpp>
#include "toolbox.h"

TEST(TestToolbox, TestHistogramPeakWidth) {
    /*
    # Test width left
    test_arr = numpy.array([0, 0, 0.5, 0.75, 0.8, 0.85, 0.9, 1])
    width = histogram_toolbox._histogram_peak_width(test_arr,
                                                    test_arr.size - 1, -1)
    assert width == 5

    # Test width right
    test_arr = test_arr[::-1]
    width = histogram_toolbox._histogram_peak_width(test_arr, 0, 1)
    assert width == 5
     */

    std::vector<float> test_arr = {0, 0, 0.5, 0.75, 0.8, 0.85, 0.9, 1};
    cv::Mat test_img(test_arr.size(), 1, CV_32FC1);
    test_img.data = (uchar*) test_arr.data();
    int width = PLImg::histogramPeakWidth(test_img, test_img.rows-1, -1);
    ASSERT_EQ(width, 5);

    test_arr = {1, 0.9, 0.85, 0.8, 0.75, 0.5, 0, 0};
    test_img = cv::Mat(test_arr.size(), 1, CV_32FC1);
    test_img.data = (uchar*) test_arr.data();
    width = PLImg::histogramPeakWidth(test_img, 0, 1);
    ASSERT_EQ(width, 5);
}

TEST(TestToolbox, TestHistogramPlateau) {

}

TEST(TestToolbox, TestImageRegionGrowing) {

}

TEST(TestToolbox, TestMedianFilter) {
    std::vector<float> test_arr = {1, 1, 3, 3, 3, 3, 2, 2, 3};
    cv::Mat test_img(3, 3, CV_32FC1);
    test_img.data = (uchar*) test_arr.data();

    auto test_img_ptr = std::make_shared<cv::Mat>(test_img);
    auto result_img = PLImg::filters::medianFilter(test_img_ptr, 1);

    std::vector<float> expected_arr = {1, 1, 3, 3, 3, 3, 2, 2, 3};
    cv::Mat expected_img(3, 3, CV_32FC1);
    expected_img.data = (uchar*) expected_arr.data();

    for(uint i = 0; i < test_img.rows; ++i) {
        for(uint j = 0; j < test_img.cols; ++j) {
            ASSERT_FLOAT_EQ(expected_img.at<float>(i, j), result_img->at<float>(i, j));
        }
    }
}

TEST(TestToolbox, TestMedianFilterMasked) {
    // Not implemented yet!
    ASSERT_TRUE(true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

