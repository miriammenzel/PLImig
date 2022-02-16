//
// Created by jreuter on 27.11.20.
//
#include <cmath>
#include "gtest/gtest.h"
#include "maskgeneration.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>

// Test function for histogram generation
void f(std::vector<float>& x) {
    for(float & i : x) {
        i = 1000.0f / (i+1.0f) + std::pow(2.0f, -i/2.0f + 4.0f);
    }
}

TEST(TestMaskgeneration, TestTRet) {
    auto x = std::vector<float>(256 * 256);
    for(ulong i = 0; i < x.size(); ++i) {
        x.at(i) = float(i) / 256.0f;
    }

    auto y = std::vector<float>(x.size());
    std::copy(x.begin(), x.end(), y.begin());
    f(y);

    float sum = 0;
    for(float i : y) {
        sum += int(i);
    }
    cv::Mat image(sum, 1, CV_32FC1);

    // Fill image with data
    unsigned current_index = 0;
    unsigned current_sum = 0;
    for(int i = 0; i < image.rows; ++i) {
        image.at<float>(i) = x.at(current_index);

        ++current_sum;
        if(current_sum >= unsigned(y.at(current_index))) {
            ++current_index;
            current_sum = 0;
        }
    }
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    auto shared_ret = std::make_shared<cv::Mat>(image);
    auto mask = PLImg::MaskGeneration(shared_ret, nullptr);
    ASSERT_FLOAT_EQ(mask.R_thres(), 0.0625f);
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
    mask.set_tref(0.3456f);
    ASSERT_FLOAT_EQ(mask.T_thres(), 0.3456f);
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
    mask.set_tback(1.0f);
    ASSERT_FLOAT_EQ(mask.T_ref(), 0.3366);
    mask.resetParameters();
}

TEST(TestMaskgeneration, TestTMax) {
    auto x = std::vector<float>(256*256);
    for(ulong i = 0; i < x.size(); ++i) {
        x.at(i) = float(i)/256.0f;
    }

    auto y = std::vector<float>(x.size());
    std::copy(x.begin(), x.end(), y.begin());
    f(y);
    std::reverse(y.begin(), y.end());

    float sum = 0;
    for(float i : y) {
        sum += i;
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
    ASSERT_FLOAT_EQ(mask.T_back(), 0.9494018f);
}

TEST(TestMaskgeneration, TestSetGet) {
    PLImg::MaskGeneration mask = PLImg::MaskGeneration();
    mask.set_tback(0.01);
    ASSERT_FLOAT_EQ(mask.T_back(), 0.01);
    mask.set_tref(0.02);
    ASSERT_FLOAT_EQ(mask.T_ref(), 0.02);
    mask.set_rthres(0.03);
    ASSERT_FLOAT_EQ(mask.R_thres(), 0.03);
    mask.set_tthres(0.04);
    ASSERT_FLOAT_EQ(mask.T_thres(), 0.04);
}

TEST(TestMaskgeneration, TestWhiteMask) {
    cv::Mat retardation(30, 30, CV_32FC1);
    cv::Mat transmittance(30, 30, CV_32FC1);

    for(int i = 0; i < retardation.rows; ++i) {
        for(int j = 0; j < retardation.cols; ++j) {
            if(j < 15) {
                retardation.at<float>(i, j) = 0.05f;
            } else {
                retardation.at<float>(i, j) = 0.15f;
            }

            if(i < 10) {
                transmittance.at<float>(i, j) = 0.05f;
            } else if (i < 20) {
                transmittance.at<float>(i, j) = 0.80f;
            } else {
                transmittance.at<float>(i, j) = 1.00f;
            }
        }
    }

    auto retPtr = std::make_shared<cv::Mat>(retardation);
    auto traPtr = std::make_shared<cv::Mat>(transmittance);
    PLImg::MaskGeneration generation(retPtr, traPtr);

    generation.set_tthres(0.5f);
    generation.set_rthres(0.1f);
    generation.set_tref(-1.0f);
    generation.set_tback(-1.0f);

    std::shared_ptr<cv::Mat> mask = generation.whiteMask();
    for(int i = 0; i < mask->rows; ++i) {
        for(int j = 0; j < mask->cols; ++j) {
            if(i < 10 | j >= 15) {
                ASSERT_TRUE(mask->at<bool>(i, j)) << "i = " << i << ", j = " << j;
            } else {
                ASSERT_FALSE(mask->at<bool>(i, j)) << "i = " << i << ", j = " << j;
            }
        }
    }
}

TEST(TestMaskgeneration, TestGrayMask) {
    cv::Mat retardation(30, 30, CV_32FC1);
    cv::Mat transmittance(30, 30, CV_32FC1);

    for(int i = 0; i < retardation.rows; ++i) {
        for(int j = 0; j < retardation.cols; ++j) {
            if(j < 15) {
                retardation.at<float>(i, j) = 0.05f;
            } else {
                retardation.at<float>(i, j) = 0.15f;
            }

            if(i <= 10) {
                transmittance.at<float>(i, j) = 0.05f;
            } else if (i < 20) {
                transmittance.at<float>(i, j) = 0.80f;
            } else {
                transmittance.at<float>(i, j) = 1.00f;
            }
        }
    }

    auto retPtr = std::make_shared<cv::Mat>(retardation);
    auto traPtr = std::make_shared<cv::Mat>(transmittance);
    PLImg::MaskGeneration generation(retPtr, traPtr);

    generation.set_tthres(0.5f);
    generation.set_rthres(0.1f);
    generation.set_tref(-1.0f);
    generation.set_tback(0.9f);

    std::shared_ptr<cv::Mat> mask = generation.grayMask();
    for(int i = 0; i < mask->rows; ++i) {
        for(int j = 0; j < mask->cols; ++j) {
            if(i > 10 && i < 20 && j < 15) {
                ASSERT_TRUE(mask->at<bool>(i, j));
            } else {
                ASSERT_FALSE(mask->at<bool>(i, j)) << "i = " << i << ", j = " << j;
            }
        }
    }
}

TEST(TestMaskgeneration, TestFullMask) {
    cv::Mat retardation(30, 30, CV_32FC1);
    cv::Mat transmittance(30, 30, CV_32FC1);

    for(int i = 0; i < retardation.rows; ++i) {
        for(int j = 0; j < retardation.cols; ++j) {
            if(j < 15) {
                retardation.at<float>(i, j) = 0.05f;
            } else {
                retardation.at<float>(i, j) = 0.15f;
            }

            if(i < 10) {
                transmittance.at<float>(i, j) = 0.05f;
            } else if (i < 20) {
                transmittance.at<float>(i, j) = 0.80f;
            } else {
                transmittance.at<float>(i, j) = 1.00f;
            }
        }
    }

    auto retPtr = std::make_shared<cv::Mat>(retardation);
    auto traPtr = std::make_shared<cv::Mat>(transmittance);
    PLImg::MaskGeneration generation(retPtr, traPtr);

    generation.set_tthres(0.5f);
    generation.set_rthres(0.1f);
    generation.set_tref(0.0f);
    generation.set_tback(0.9f);

    auto whiteMask = generation.whiteMask();
    auto grayMask = generation.grayMask();
    auto fullMask = generation.fullMask();

    for(int i = 0; i < fullMask->total(); ++i) {
        if(whiteMask->at<bool>(i)) {
            ASSERT_EQ(fullMask->at<unsigned char>(i), 200);
        } else if(grayMask->at<bool>(i)) {
            ASSERT_EQ(fullMask->at<unsigned char>(i), 100);
        } else {
            ASSERT_EQ(fullMask->at<unsigned char>(i), 0);
        }
    }
}

TEST(TestMaskgeneration, TestBlurredMask) {

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
