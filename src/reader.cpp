//
// Created by jreuter on 25.11.20.
//

#include "reader.h"

bool PLImg::fileExists(const std::string& filename) {
    struct stat buffer{};
    return (stat (filename.c_str(), &buffer) == 0);
}

cv::Mat PLImg::imread(const std::string& filename, const std::string& dataset) {
    cv::Mat image;
    if(fileExists(filename)) {
        if(filename.substr(filename.size()-2) == "h5") {
            cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( filename );
            h5io->dsread(image, dataset);
            h5io->close();
        } else if(filename.substr(filename.size()-3) == "nii"){
            //image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            throw std::filesystem::filesystem_error(".nii files aren't supported yet.", std::error_code(11, std::generic_category()));
        } else {
            image = cv::imread(filename, cv::IMREAD_ANYDEPTH);
        }
    } else {
        throw std::filesystem::filesystem_error("File not found: " + filename, std::error_code(10, std::generic_category()));
    }
    return image;
}
