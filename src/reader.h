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

#ifndef PLIMG_READER_H
#define PLIMG_READER_H

#include <filesystem>
#include <hdf5.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <tiffio.h>
#include <nifti/nifti1_io.h>

namespace PLImg {
    class reader {
    public:
        /**
         * @brief fileExists
         * @param filename
         * @return
         */
        static bool fileExists(const std::string& filename);
        /**
         * @brief imread
         * @param filename
         * @param dataset
         * @return
         */
        static cv::Mat imread(const std::string& filename, const std::string& dataset="/Image");
    private:
        /**
         * @brief readHDF5
         * @param filename
         * @param dataset
         * @return
         */
        static cv::Mat readHDF5(const std::string& filename, const std::string& dataset="/Image");
        /**
         * @brief readTiff
         * @param filename
         * @return
         */
        static cv::Mat readTiff(const std::string& filename);
        /**
         * @brief readNIFTI
         * @param filename
         * @return
         */
        static cv::Mat readNIFTI(const std::string& filename);
    };

}


#endif //PLIMG_READER_H
