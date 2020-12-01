//
// Created by jreuter on 27.11.20.
//

#include "gtest/gtest.h"
#include "H5Cpp.h"
#include <opencv2/core.hpp>
#include "reader.h"

TEST(ReaderTest, TestFileExists) {
    ASSERT_TRUE(PLImg::reader::fileExists("../../tests/files/demo.h5"));
    ASSERT_TRUE(PLImg::reader::fileExists("../../tests/files/demo.tiff"));
    ASSERT_TRUE(PLImg::reader::fileExists("../../tests/files/demo.nii"));
}

TEST(ReaderTest, TestImageReadHDF5) {
    auto image = PLImg::reader::imread("../../tests/files/demo.h5", "/pyramid/06");
    ASSERT_EQ(image.rows, 195);
    ASSERT_EQ(image.cols, 150);
    ASSERT_EQ(image.type(), CV_32FC1);
}

TEST(ReaderTest, TestImageReadTiff) {
    auto image = PLImg::reader::imread("../../tests/files/demo.tiff");
    ASSERT_EQ(image.rows, 195);
    ASSERT_EQ(image.cols, 150);
    ASSERT_EQ(image.type(), CV_32FC1);
}

TEST(ReaderTest, TestImageReadNIFTI) {
    auto image = PLImg::reader::imread("../../tests/files/demo.nii");
    ASSERT_EQ(image.rows, 195);
    ASSERT_EQ(image.cols, 150);
    ASSERT_EQ(image.type(), CV_32FC1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}