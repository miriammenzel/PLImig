//
// Created by jreuter on 27.11.20.
//

#include "gtest/gtest.h"
#include "writer.h"
#include "H5Cpp.h"

TEST(WriterTest, TestEmpty) {
    PLImg::HDF5Writer writer;
    ASSERT_EQ(writer.path(), "");
}

TEST(WriterTest, TestWriteAttributes) {
    PLImg::HDF5Writer writer;
    float tra = 0;
    float ret = 0.1;
    float min = 0.2;
    float max = 0.3;
    writer.set_path("output/writer_test_1.h5");
    writer.write_attributes("/", tra, ret, min, max);
    writer.close();

    auto file = H5::H5File("output/writer_test_1.h5", H5F_ACC_RDONLY);

    double rtra, rret, rmin, rmax;

    auto attr = file.openAttribute("t_tra");
    auto datatype = attr.getDataType();
    attr.read(datatype, &rtra);
    attr.close();
    ASSERT_FLOAT_EQ(rtra, tra);

    attr = file.openAttribute("t_ret");
    datatype = attr.getDataType();
    attr.read(datatype, &rret);
    attr.close();
    ASSERT_FLOAT_EQ(rret, ret);

    attr = file.openAttribute("t_min");
    datatype = attr.getDataType();
    attr.read(datatype, &rmin);
    attr.close();
    ASSERT_FLOAT_EQ(rmin, min);

    attr = file.openAttribute("t_max");
    datatype = attr.getDataType();
    attr.read(datatype, &rmax);
    attr.close();
    ASSERT_FLOAT_EQ(rmax, max);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}