//
// Created by jreuter on 27.11.20.
//
#include "gtest/gtest.h"
#include <cmath>
#include <memory>
#include <random>
#include "toolbox.h"

// Test function for histogram generation
void f(std::vector<float>& x) {
    for(float & i : x) {
        i = 10000.0f / (i+1.0f) + 1000.0f * std::pow(2.0f, -i/2.0f + 4.0f);
    }
}

TEST(TestToolbox, TestRunCudaChecks) {
    ASSERT_TRUE(PLImg::cuda::runCUDAchecks());
}

TEST(TestToolbox, TestHistogramPeakWidth) {
    std::vector<float> test_arr = {0, 0, 0.5, 0.75, 0.8, 0.85, 0.9, 1};
    cv::Mat test_img(test_arr.size(), 1, CV_32FC1);
    test_img.data = (uchar*) test_arr.data();
    int width = PLImg::Histogram::peakWidth(test_img, test_img.rows-1, -1);
    ASSERT_EQ(width, 5);

    test_arr = {1, 0.9, 0.85, 0.8, 0.75, 0.5, 0, 0};
    test_img = cv::Mat(test_arr.size(), 1, CV_32FC1);
    test_img.data = (uchar*) test_arr.data();
    width = PLImg::Histogram::peakWidth(test_img, 0, 1);
    ASSERT_EQ(width, 5);
}

TEST(TestToolbox, TestHistogramPlateauLeft) {
    auto x = std::vector<float>(256 / 0.01);
    for(unsigned long i = 0; i < x.size(); ++i) {
        x.at(i) = float(i) * 0.01f;
    }

    auto y = std::vector<float>(256 / 0.01);
    std::copy(x.begin(), x.end(), y.begin());
    f(y);

    int sum = int(std::accumulate(y.begin(), y.end(), 0.0f));
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
    int histSize = MAX_NUMBER_OF_BINS;

    // Generate histogram
    cv::Mat hist(MAX_NUMBER_OF_BINS, 1, CV_32FC1);
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    cv::normalize(hist, hist, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);

    cv::Mat curvatureVal = PLImg::Histogram::curvature(hist, 0, 1);
    int result = std::max_element(curvatureVal.begin<float>(), curvatureVal.begin<float>() + 20) - curvatureVal.begin<float>();
    ASSERT_EQ(result, 15);
}

TEST(TestToolbox, TestHistogramPlateauRight) {
    auto x = std::vector<float>(256 / 0.01);
    for(unsigned long i = 0; i < x.size(); ++i) {
        x.at(i) = i * 0.01;
    }

    auto y = std::vector<float>(256 / 0.01);
    std::copy(x.begin(), x.end(), y.begin());
    f(y);

    int sum = int(std::accumulate(y.begin(), y.end(), 0.0f));
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
    int histSize = MAX_NUMBER_OF_BINS;

    // Generate histogram
    cv::Mat hist(histSize, 1, CV_32FC1);
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    std::reverse(hist.begin<float>(), hist.end<float>());

    cv::Mat curvatureVal = PLImg::Histogram::curvature(hist, 0, 1);
    int result = std::max_element(curvatureVal.end<float>() - 30, curvatureVal.end<float>()) - curvatureVal.begin<float>();
    ASSERT_EQ(result, 240);
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

    cv::Mat mask = PLImg::cuda::labeling::largestAreaConnectedComponents(test_retardation, cv::Mat());
    for(uint i = 11; i < 20; ++i) {
        for(uint j = 11; j < 15; ++j) {
            ASSERT_TRUE(mask.at<bool>(i, j));
        }
    }
    ASSERT_FLOAT_EQ(cv::mean(test_transmittance, mask)[0], 0.3456);
}

TEST(TestToolbox, TestMedianFilter) {
    auto testImage = cv::imread("../../tests/files/median_filter/median_input.tiff", cv::IMREAD_ANYDEPTH);
    auto expectedResult = cv::imread("../../tests/files/median_filter/median_expected_result.tiff", cv::IMREAD_ANYDEPTH);

    auto testImagePtr = std::make_shared<cv::Mat>(testImage);
    auto medianFilterPtr = PLImg::cuda::filters::medianFilter(testImagePtr);

    for(int i = 0; i < expectedResult.rows; ++i) {
        for(int j = 0; j < expectedResult.cols; ++j) {
            ASSERT_FLOAT_EQ(medianFilterPtr->at<float>(i, j), expectedResult.at<float>(i, j));
        }
    }
}

TEST(TestToolbox, TestMedianFilterMasked) {
    // Not implemented yet!
    ASSERT_TRUE(true);
}

TEST(TestToolbox, TestConnectedComponents) {
    uint maxNumber;
    cv::Mat exampleMask = (cv::Mat1s(7, 11) <<
            1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0,
            1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
            0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
            0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1);
    exampleMask.convertTo(exampleMask, CV_8UC1);

    cv::Mat resultMask = (cv::Mat1s(7, 11) <<
            2, 2, 2, 0, 0, 0, 5, 0, 1, 1, 0,
            2, 0, 0, 0, 4, 0, 5, 5, 0, 0, 0,
            0, 0, 4, 4, 4, 0, 5, 0, 3, 3, 3,
            0, 0, 4, 4, 4, 0, 5, 0, 0, 0, 0,
            6, 0, 0, 0, 0, 0, 5, 0, 7, 0, 7,
            6, 6, 6, 6, 6, 0, 0, 0, 7, 7, 7,
            0, 0, 6, 6, 6, 0, 0, 0, 7, 7, 7);
    resultMask.convertTo(resultMask, CV_32SC1);
    cv::Mat result = PLImg::cuda::raw::labeling::CUDAConnectedComponents(exampleMask, &maxNumber);
    cv::imwrite("output.tiff", result);

    for(int x = 0; x < resultMask.cols; ++x) {
        for(int y = 0; y < resultMask.rows; ++y) {
            ASSERT_EQ(result.at<int>(y, x), resultMask.at<int>(y, x)) << x << "," << y;
        }
    }
    ASSERT_EQ(maxNumber, 7);
}

TEST(TestToolbox, TestConnectedComponentsUF) {
    uint maxNumber;
    cv::Mat exampleMask = (cv::Mat1s(7, 11) <<
            1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0,
            1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
            0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
            0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1);
    exampleMask.convertTo(exampleMask, CV_8UC1);

    cv::Mat resultMask = (cv::Mat1s(7, 11) <<
            1, 1, 1, 0, 0, 0, 2, 0, 2, 2, 0,
            1, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0,
            0, 0, 3, 3, 3, 0, 2, 0, 2, 2, 2,
            0, 0, 3, 3, 3, 0, 2, 0, 0, 0, 0,
            4, 0, 0, 0, 0, 0, 2, 0, 5, 0, 5,
            4, 4, 4, 4, 4, 0, 0, 0, 5, 5, 5,
            0, 0, 4, 4, 4, 0, 0, 0, 5, 5, 5);
    resultMask.convertTo(resultMask, CV_32SC1);
    cv::Mat result = PLImg::cuda::raw::labeling::CUDAConnectedComponentsUF(exampleMask, &maxNumber);

    for(int x = 0; x < resultMask.cols; ++x) {
        for(int y = 0; y < resultMask.rows; ++y) {
            ASSERT_EQ(result.at<int>(y, x), resultMask.at<int>(y, x)) << x << "," << y;
        }
    }
    ASSERT_EQ(maxNumber, 6);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

