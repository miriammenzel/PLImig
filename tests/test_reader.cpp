//
// Created by jreuter on 27.11.20.
//

#include "gtest/gtest.h"
#include "H5Cpp.h"
#include <opencv2/core.hpp>
#include "reader.h"

TEST(ReaderTest, TestFileExists) {
    ASSERT_TRUE(PLImg::fileExists("../../tests/files/demo.h5"));
    ASSERT_TRUE(PLImg::fileExists("../../tests/files/demo.tiff"));
    ASSERT_TRUE(PLImg::fileExists("../../tests/files/demo.nii"));
}

TEST(ReaderTest, TestImageRead) {
    auto image = PLImg::imread("../../tests/files/demo.h5", "/pyramid/06");
    ASSERT_EQ(image.rows, 195);
    ASSERT_EQ(image.cols, 150);
    ASSERT_EQ(image.type(), CV_32FC1);

    image = PLImg::imread("../../tests/files/demo.tiff", "/pyramid/06");
    ASSERT_EQ(image.rows, 195);
    ASSERT_EQ(image.cols, 150);
    ASSERT_EQ(image.type(), CV_32FC1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}