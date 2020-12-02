//
// Created by jreuter on 27.11.20.
//
#include <cmath>
#include "gtest/gtest.h"
#include "maskgeneration.h"
#include <memory>
#include <opencv2/core.hpp>
#include <random>
#include "toolbox.h"

// Test function for histogram generation
void f(std::vector<float>& x) {
    for(float & i : x) {
        i = 11.0f / (i + 1.0f) + std::pow(4.0f, -i + 4.0f);
    }
}

TEST(TestMaskgeneration, TestTRet) {
    auto x = std::vector<float>(256*256);
    for(int i = 0; i < x.size(); ++i) {
        x.at(i) = float(i)/256.0f;
    }

    auto y = std::vector<float>(x.size());
    std::copy(x.begin(), x.end(), y.begin());
    f(y);

    float sum = 0;
    for(float i : y) {
        sum += std::ceil(i);
    }
    cv::Mat image(sum, 1, CV_32FC1);

    // Fill image with data
    unsigned current_index = 0;
    unsigned current_sum = 0;
    for(int i = 0; i < image.rows; ++i) {
        image.at<float>(i) = x.at(current_index);

        ++current_sum;
        if(current_sum > unsigned(y.at(current_index))) {
            ++current_index;
            current_sum = 0;
        }
    }
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

    auto shared_ret = std::make_shared<cv::Mat>(image);
    auto mask = PLImg::MaskGeneration(shared_ret, nullptr);
    ASSERT_FLOAT_EQ(mask.tRet(), 4.0f * (1.0f/256.0f));
}

TEST(TestMaskgeneration, TestTTra) {
    cv::Mat test_retardation(100, 100, CV_32FC1);
    cv::Mat test_transmittance(100, 100, CV_32FC1);

    for(uint i = 0; i < 100; ++i) {
        for(uint j = 0; j < 100; ++j) {
            if(i > 10 && i < 20 && j > 10 && j < 15) {
                test_retardation.at<float>(i, j) = 0.99f;
                test_transmittance.at<float>(i, j) = 0.3456f;
            } else {
                test_retardation.at<float>(i, j) = 0.0f;
                test_transmittance.at<float>(i, j) = 0.0f;
            }
        }
    }

    auto shared_ret = std::make_shared<cv::Mat>(test_retardation);
    auto shared_tra = std::make_shared<cv::Mat>(test_transmittance);
    auto mask = PLImg::MaskGeneration(shared_ret, shared_tra);
    ASSERT_FLOAT_EQ(mask.tTra(), 0.3456);
}

TEST(TestMaskgeneration, TestTMin) {
    cv::Mat test_retardation(100, 100, CV_32FC1);
    cv::Mat test_transmittance(100, 100, CV_32FC1);

    for(uint i = 0; i < 100; ++i) {
        for(uint j = 0; j < 100; ++j) {
            if(i > 10 && i < 20 && j > 10 && j < 15) {
                test_retardation.at<float>(i, j) = 0.99f;
                test_transmittance.at<float>(i, j) = 0.3366;
            } else {
                test_retardation.at<float>(i, j) = 0.0f;
                test_transmittance.at<float>(i, j) = 0.0f;
            }
        }
    }

    auto shared_ret = std::make_shared<cv::Mat>(test_retardation);
    auto shared_tra = std::make_shared<cv::Mat>(test_transmittance);
    auto mask = PLImg::MaskGeneration(shared_ret, shared_tra);
    ASSERT_FLOAT_EQ(mask.tMin(), 0.3366);
}

TEST(TestMaskgeneration, TestTMax) {
    auto x = std::vector<float>(256*256);
    for(int i = 0; i < x.size(); ++i) {
        x.at(i) = float(i)/256.0f;
    }

    auto y = std::vector<float>(x.size());
    std::copy(x.begin(), x.end(), y.begin());
    f(y);
    std::reverse(y.begin(), y.end());

    float sum = 0;
    for(float i : y) {
        sum += std::ceil(i);
    }
    cv::Mat image(sum, 1, CV_32FC1);

    // Fill image with data
    unsigned current_index = 0;
    unsigned current_sum = 0;
    for(int i = 0; i < image.rows; ++i) {
        image.at<float>(i) = x.at(current_index);

        ++current_sum;
        if(current_sum > unsigned(y.at(current_index))) {
            ++current_index;
            current_sum = 0;
        }
    }
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    auto shared_tra = std::make_shared<cv::Mat>(image);
    auto mask = PLImg::MaskGeneration(nullptr, shared_tra);
    ASSERT_FLOAT_EQ(mask.tMax(), 250.0f * (1.0f/256.0f));
}

TEST(TestMaskgeneration, TestSetGet) {
    PLImg::MaskGeneration mask = PLImg::MaskGeneration();
    mask.set_tMax(0.01);
    ASSERT_FLOAT_EQ(mask.tMax(), 0.01);
    mask.set_tMin(0.02);
    ASSERT_FLOAT_EQ(mask.tMin(), 0.02);
    mask.set_tRet(0.03);
    ASSERT_FLOAT_EQ(mask.tRet(), 0.03);
    mask.set_tTra(0.04);
    ASSERT_FLOAT_EQ(mask.tTra(), 0.04);
}

TEST(TestMaskgeneration, TestWhiteMask) {

}

TEST(TestMaskgeneration, TestGrayMask) {

}

TEST(TestMaskgeneration, TestFullMask) {

}

TEST(TestMaskgeneration, TestNoNerveFiberMask) {

}

TEST(TestMaskgeneration, TestBlurredMask) {

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}