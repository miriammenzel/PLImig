//
// Created by jreuter on 27.11.20.
//
#include "gtest/gtest.h"
#include "maskgeneration.h"

TEST(TestMaskgeneration, TestTRet) {

}

TEST(TestMaskgeneration, TestTTra) {

}

TEST(TestMaskgeneration, TestTMin) {

}

TEST(TestMaskgeneration, TestTMax) {

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