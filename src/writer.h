/*
    MIT License

    Copyright (c) 2020 Forschungszentrum Jülich / Jan André Reuter.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
 */

#ifndef PLIMG_WRITER_H
#define PLIMG_WRITER_H

#include <filesystem>
#include <H5Cpp.h>
#include <opencv2/core.hpp>
#include "reader.h"

namespace PLImg {
    /**
     * @brief The HDF5Writer class
     */
    class HDF5Writer {
    public:
        /**
         * @brief HDF5Writer
         */
        HDF5Writer();
        /**
         * @brief path
         * @return
         */
        std::string path();
        /**
         * @brief set_path
         * @param filename
         */
        void set_path(const std::string& filename);
        /**
         * @brief write_attributes
         * @param dataset
         * @param t_tra
         * @param t_ret
         * @param t_min
         * @param t_max
         */
        void write_attributes(std::string dataset, float t_tra, float t_ret, float t_min, float t_max);
        /**
         * @brief write_dataset
         * @param dataset
         * @param image
         */
        void write_dataset(const std::string& dataset, const cv::Mat& image);
        /**
         * @brief create_group
         * @param group
         */
        void create_group(const std::string& group);
        /**
         * @brief close
         */
        void close();
    private:
        /**
         * @brief open
         */
        void open();
        /**
         * @brief createDirectoriesIfMissing
         * @param filename
         */
        static void createDirectoriesIfMissing(const std::string& filename);

        ///
        std::string m_filename;
        ///
        H5::H5File m_hdf5file;
    };
}


#endif //PLIMG_WRITER_H
