//
// Created by jreuter on 27.11.20.
//
#include "gtest/gtest.h"
#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <random>
#include "toolbox.h"

// Test function for histogram generation
void f(std::vector<float>& x) {
    for(float & i : x) {
        i = 1000.0f * exp(-2*i-5) + 1;
    }
}

TEST(TestToolbox, TestRunCudaChecks) {
    ASSERT_TRUE(PLImg::cuda::runCUDAchecks());
}

TEST(TestToolbox, TestHistogramPeakWidth) {
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

TEST(TestToolbox, TestHistogramPlateauLeft) {
    auto x = std::vector<float>(128 / 0.01);
    for(int i = 0; i < x.size(); ++i) {
        x.at(i) = i * 0.01;
    }

    auto y = std::vector<float>(128 / 0.01);
    std::copy(x.begin(), x.end(), y.begin());
    f(y);

    int sum = int(std::accumulate(y.begin(), y.end(), 0));
    cv::Mat image(sum, 1, CV_32FC1);

    // Fill image with random data
    unsigned current_index = 0;
    float current_sum = 0;
    for(int i = 0; i < image.rows; ++i) {
        image.at<float>(i) = x.at(current_index);
        ++current_sum;
        if(current_sum > y.at(current_index)) {
            ++current_index;
            current_sum = 0;
        }
    }
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

    // Generate histogram
    int channels[] = {0};
    float histBounds[] = {0.0f, 1.0f};
    const float* histRange = { histBounds };
    int histSize = NUMBER_OF_BINS;

    // Generate histogram
    cv::Mat hist;
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    float result = PLImg::histogramPlateau(hist, 0, 1, 1, 1, NUMBER_OF_BINS/2);
    ASSERT_FLOAT_EQ(result, 4.0f * (1.0f/256.0f));
}

TEST(TestToolbox, TestHistogramPlateauRight) {
    auto x = std::vector<float>(128 / 0.01);
    for(int i = 0; i < x.size(); ++i) {
        x.at(i) = i * 0.01;
    }

    auto y = std::vector<float>(128 / 0.01);
    std::copy(x.begin(), x.end(), y.begin());
    f(y);

    int sum = int(std::accumulate(y.begin(), y.end(), 0));
    cv::Mat image(sum, 1, CV_32FC1);

    // Fill image with random data
    unsigned current_index = 0;
    float current_sum = 0;
    for(int i = 0; i < image.rows; ++i) {
        image.at<float>(i) = x.at(current_index);
        ++current_sum;
        if(current_sum > y.at(current_index)) {
            ++current_index;
            current_sum = 0;
        }
    }
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

    // Generate histogram
    int channels[] = {0};
    float histBounds[] = {0.0f, 1.0f};
    const float* histRange = { histBounds };
    int histSize = NUMBER_OF_BINS;

    // Generate histogram
    cv::Mat hist;
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    std::reverse(hist.begin<float>(), hist.end<float>());

    float result = PLImg::histogramPlateau(hist, 0, 1, -1, NUMBER_OF_BINS/2, NUMBER_OF_BINS);
    ASSERT_FLOAT_EQ(result, 1.0f - 5.0f * (1.0f/256.0f));
}

TEST(TestToolbox, TestImageRegionGrowing) {
    cv::Mat test_retardation(100, 100, CV_32FC1);
    cv::Mat test_transmittance(100, 100, CV_32FC1);

    for(uint i = 0; i < 100; ++i) {
        for(uint j = 0; j < 100; ++j) {
            if(i > 10 && i < 20 && j > 10 && j < 15) {
                test_retardation.at<float>(i, j) = 0.975f;
                test_transmittance.at<float>(i, j) = 0.3456f;
            } else {
                test_retardation.at<float>(i, j) = 0.0f;
                test_transmittance.at<float>(i, j) = 0.0f;
            }
        }
    }

    cv::Mat mask = PLImg::imageRegionGrowing(test_retardation);
    for(uint i = 11; i < 20; ++i) {
        for(uint j = 11; j < 15; ++j) {
            ASSERT_TRUE(mask.at<bool>(i, j));
        }
    }
    ASSERT_FLOAT_EQ(cv::mean(test_transmittance, mask)[0], 0.3456);
}

TEST(TestToolbox, TestMedianFilter) {
    std::vector<float> test_arr = {1, 1, 3, 3, 3, 3, 2, 2, 3};
    cv::Mat test_img(3, 3, CV_32FC1);
    test_img.data = (uchar*) test_arr.data();

    auto test_img_ptr = std::make_shared<cv::Mat>(test_img);
    auto result_img = PLImg::cuda::filters::medianFilter(test_img_ptr);

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

