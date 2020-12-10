//
// Created by jreuter on 25.11.20.
//

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
        static inline bool fileExists(const std::string& filename);
        static cv::Mat imread(const std::string& filename, const std::string& dataset="/Image");
    private:
        static cv::Mat readHDF5(const std::string& filename, const std::string& dataset="/Image");
        static cv::Mat readTiff(const std::string& filename);
        static cv::Mat readNIFTI(const std::string& filename);
    };

}


#endif //PLIMG_READER_H
