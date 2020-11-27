//
// Created by jreuter on 25.11.20.
//

#include "reader.h"

bool PLImg::fileExists(std::string filename) {
    struct stat buffer{};
    return (stat (filename.c_str(), &buffer) == 0);
}

cv::Mat PLImg::imread(std::string filename, std::string dataset) {
    cv::Mat image;
    if(fileExists(filename)) {
        if(filename.substr(filename.size()-2) == "h5") {
            cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( filename );
            h5io->dsread(image, dataset);
            h5io->close();
        } else {
            image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        }
    } else {
        throw std::filesystem::filesystem_error("File not found: " + filename, std::error_code(10, std::generic_category()));
    }
    return image;
}
