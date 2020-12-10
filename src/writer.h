//
// Created by jreuter on 25.11.20.
//

#ifndef PLIMG_WRITER_H
#define PLIMG_WRITER_H

#include <filesystem>
#include <hdf5.h>
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
        void switchHDF5ErrorHandling(bool on);

        std::string m_filename;
        hid_t m_hdf5File;
        H5E_auto2_t errorFunction;
        void* errorFunctionData;
    };
}


#endif //PLIMG_WRITER_H
