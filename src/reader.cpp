//
// Created by jreuter on 25.11.20.
//

#include "reader.h"

bool PLImg::Reader::fileExists(const std::string& filename) {
    struct stat buffer{};
    return (stat (filename.c_str(), &buffer) == 0);
}

cv::Mat PLImg::Reader::imread(const std::string& filename, const std::string& dataset) {
    // Check if file exists
    if(fileExists(filename)) {
        // Opening the file has to be handeled differently depending on the file ending.
        // This will be done here.
        if(filename.substr(filename.size()-2) == "h5") {
            return readHDF5(filename, dataset);
        } else if(filename.substr(filename.size()-3) == "nii" || filename.substr(filename.size()-6) == "nii.gz"){
            return readNIFTI(filename);
        } else {
            return readTiff(filename);
        }
    } else {
        throw std::filesystem::filesystem_error("File not found: " + filename, std::error_code(10, std::generic_category()));
    }
}

cv::Mat PLImg::Reader::readHDF5(const std::string &filename, const std::string &dataset) {
    hid_t file, dspace, dset;
    hsize_t dims[2];
    // Open file read only
    file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    // Open dataset
    dset = H5Dopen(file, dataset.c_str(), H5P_DEFAULT);
    // Get dataspace
    dspace = H5Dget_space(dset);
    // Get image dimensions
    H5Sget_simple_extent_dims(dspace, dims, nullptr);

    // OpenCV does use other names and integers for its own datatype handling.
    // Check the HDF5 type and convert it to a valid OpenCV mat type.
    hid_t type = H5Dget_type(dset);
    int matType;
    if(H5Tequal(type, H5T_NATIVE_UCHAR)) {
        matType = CV_8UC1;
    } else if(H5Tequal(type, H5T_NATIVE_FLOAT)) {
        matType = CV_32FC1;
    } else if(H5Tequal(type, H5T_NATIVE_INT)) {
        matType = CV_32SC1;
    } else {
        throw std::runtime_error("Datatype is currently not supported. Please contact the maintainer of the program!");
    }
    // Create OpenCV mat and copy content from dataset to mat
    cv::Mat image(dims[0], dims[1], matType);
    H5Dread(dset, type, dspace, H5S_ALL, H5S_ALL, image.data);

    H5Tclose(type);
    H5Sclose(dspace);
    H5Dclose(dset);
    H5Fclose(file);

    return image;
}

cv::Mat PLImg::Reader::readNIFTI(const std::string &filename) {
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

cv::Mat PLImg::Reader::readTiff(const std::string &filename) {
    return cv::imread(filename, cv::IMREAD_ANYDEPTH);
}
