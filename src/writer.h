//
// Created by jreuter on 25.11.20.
//

#ifndef PLIMG_WRITER_H
#define PLIMG_WRITER_H

#include <filesystem>
#include <H5Cpp.h>
#include <opencv2/core.hpp>
#include "reader.h"

namespace PLImg {
    class HDF5Writer {
    public:
        HDF5Writer();
        std::string path();
        void set_path(const std::string& filename);
        void write_attributes(std::string dataset, float t_tra, float t_ret, float t_min, float t_max);
        void write_dataset(const std::string& dataset, const cv::Mat& image);
        void create_group(const std::string& group);
        void close();
    private:
        void open();
        static void createDirectoriesIfMissing(const std::string& filename);

        std::string m_filename;
        H5::H5File m_hdf5file;
    };
}


#endif //PLIMG_WRITER_H
