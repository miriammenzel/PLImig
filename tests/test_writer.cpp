//
// Created by jreuter on 27.11.20.
//

#include "gtest/gtest.h"
#include "H5Cpp.h"
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>
#include "writer.h"


TEST(WriterTest, TestEmpty) {
    PLImg::HDF5Writer writer;
    ASSERT_EQ(writer.path(), "");
}

TEST(WriterTest, TestWriteAttributes) {
    PLImg::HDF5Writer writer;
    float tra = 0;
    double ret = 0.1;
    std::string min = "0.2";
    int max = 3;
    writer.set_path("output/writer_test_1.h5");
    writer.write_attribute("/", "t_tra", tra);
    writer.write_attribute("/", "t_ret", ret);
    writer.write_attribute("/", "t_min", min);
    writer.write_attribute("/", "t_max", max);
    writer.close();

    auto file = H5::H5File("output/writer_test_1.h5", H5F_ACC_RDONLY);

    float rtra;
    double rret;
    std::string rmin;
    int rmax;

    auto attr = file.openAttribute("t_tra");
    auto datatype = attr.getDataType();
    attr.read(datatype, &rtra);
    attr.close();
    ASSERT_FLOAT_EQ(rtra, tra);

    attr = file.openAttribute("t_ret");
    datatype = attr.getDataType();
    attr.read(datatype, &rret);
    attr.close();
    ASSERT_DOUBLE_EQ(rret, ret);

    attr = file.openAttribute("t_min");
    datatype = attr.getDataType();
    attr.read(datatype, &rmin);
    attr.close();
    ASSERT_STRCASEEQ(rmin.c_str(), min.c_str());

    attr = file.openAttribute("t_max");
    datatype = attr.getDataType();
    attr.read(datatype, &rmax);
    attr.close();
    ASSERT_EQ(rmax, max);
}

TEST(WriterTest, TestWriteDataset) {
    PLImg::HDF5Writer writer;

    cv::Mat testMat(10, 10, CV_32FC1);
    for(unsigned i = 0; i < testMat.rows; ++i) {
        for(unsigned j = 0; j < testMat.cols; ++j) {
            testMat.at<float>(i, j) = i * testMat.cols + j;
        }
    }

    writer.set_path("output/writer_test_2.h5");
    writer.write_dataset("test_write_dataset", testMat);
    writer.close();

    H5::H5File file("output/writer_test_2.h5", H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("test_write_dataset");
    H5::DataSpace space = dataset.getSpace();
    H5::DataType type = dataset.getDataType();
    hsize_t dims[2];
    space.getSimpleExtentDims(dims);

    cv::Mat readMat(dims[0], dims[1], CV_32FC1);
    dataset.read(readMat.data, type, space);

    for(unsigned i = 0; i < testMat.rows; ++i) {
        for(unsigned j = 0; j < testMat.cols; ++j) {
            ASSERT_FLOAT_EQ(testMat.at<float>(i, j), readMat.at<float>(i, j));
        }
    }
}

TEST(WriterTest, TestCreateGroup) {
    PLImg::HDF5Writer writer;
    writer.set_path("output/writer_test_3.h5");
    writer.create_group("/demogroup");
    auto file = cv::hdf::open( "output/writer_test_3.h5" );
    ASSERT_TRUE(file->hlexists("/demogroup"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}