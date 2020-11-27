//
// Created by jreuter on 25.11.20.
//

#ifndef PLIMG_READER_H
#define PLIMG_READER_H

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/hdf/hdf5.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>

namespace PLImg {
    inline bool fileExists(std::string filename);
    cv::Mat imread(std::string filename, std::string dataset="/Image");
}


#endif //PLIMG_READER_H
