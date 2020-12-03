//
// Created by jreuter on 25.11.20.
//

#include "reader.h"

inline bool PLImg::reader::fileExists(const std::string& filename) {
    struct stat buffer{};
    return (stat (filename.c_str(), &buffer) == 0);
}

cv::Mat PLImg::reader::imread(const std::string& filename, const std::string& dataset) {
    if(fileExists(filename)) {
        if(filename.substr(filename.size()-2) == "h5") {
            return readHDF5(filename, dataset);
        } else if(filename.substr(filename.size()-3) == "nii"){
            return readNIFTI(filename);
        } else {
            return readTiff(filename);
        }
    } else {
        throw std::filesystem::filesystem_error("File not found: " + filename, std::error_code(10, std::generic_category()));
    }
}

cv::Mat PLImg::reader::readHDF5(const std::string &filename, const std::string &dataset) {
    cv::Mat image;
    cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( filename );
    h5io->dsread(image, dataset);
    h5io->close();
    return image;
}

cv::Mat PLImg::reader::readNIFTI(const std::string &filename) {
    nifti_image * img = nifti_image_read(filename.c_str(), 1);
    // Get image dimensions
    uint width = img->nx;
    uint height = img->ny;
    // Convert NIFTI datatype to OpenCV datatype
    uint datatype = img->datatype;
    uint cv_type;
    switch(datatype) {
        case 16:
            cv_type = CV_32FC1;
            break;
        case 8:
            cv_type = CV_32SC1;
            break;
        case 4:
            cv_type = CV_16SC1;
            break;
        case 2:
            cv_type = CV_8SC1;
        default:
            throw std::runtime_error("Did expect 32-bit floating point or 8/16/32-bit integer image!");
    }
    // Create OpenCV image with the image data
    cv::Mat image(height, width, cv_type);
    image.data = (uchar*) img->data;
    return image;
}

cv::Mat PLImg::reader::readTiff(const std::string &filename) {
    return cv::imread(filename, cv::IMREAD_ANYDEPTH);
}
