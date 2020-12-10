//
// Created by jreuter on 25.11.20.
//

#include "writer.h"
#include <iostream>

PLImg::HDF5Writer::HDF5Writer() {
    /* Save old error handler */
    H5Eget_auto(H5E_DEFAULT, &errorFunction, &errorFunctionData);
}

std::string PLImg::HDF5Writer::path() {
    return this->m_filename;
}

void PLImg::HDF5Writer::set_path(const std::string& filename) {
    if(this->m_filename != filename) {
        this->m_filename = filename;
        this->open();
    }
}

void PLImg::HDF5Writer::write_attributes(std::string dataset, float t_tra, float t_ret, float t_min, float t_max) {
    while(!dataset.empty() && dataset.at(dataset.size()-1) == '/') {
        dataset = dataset.substr(0, dataset.size()-1);
    }

    hid_t attrSpace;
    hid_t attrHandle;
    hid_t groupHandle = H5Gopen1(m_hdf5File, "/");
    switchHDF5ErrorHandling(false);

    H5Adelete(m_hdf5File, (dataset+"t_tra").c_str());
    attrSpace = H5Screate(H5S_SCALAR);
    attrHandle = H5Acreate1(groupHandle, "t_tra", H5T_IEEE_F32LE, attrSpace, H5P_DEFAULT);
    H5Awrite(attrHandle, H5T_IEEE_F32LE, &t_tra);
    H5Sclose(attrSpace);
    H5Aclose(attrHandle);

    H5Adelete(m_hdf5File, (dataset+"t_ret").c_str());
    attrSpace = H5Screate(H5S_SCALAR);
    attrHandle = H5Acreate1(groupHandle, "t_ret", H5T_IEEE_F32LE, attrSpace, H5P_DEFAULT);
    H5Awrite(attrHandle, H5T_IEEE_F32LE, &t_ret);
    H5Sclose(attrSpace);
    H5Aclose(attrHandle);

    H5Adelete(m_hdf5File, (dataset+"t_min").c_str());
    attrSpace = H5Screate(H5S_SCALAR);
    attrHandle = H5Acreate1(groupHandle, "t_min", H5T_IEEE_F32LE, attrSpace, H5P_DEFAULT);
    H5Awrite(attrHandle, H5T_IEEE_F32LE, &t_min);
    H5Sclose(attrSpace);
    H5Aclose(attrHandle);

    H5Adelete(m_hdf5File, (dataset+"t_max").c_str());
    attrSpace = H5Screate(H5S_SCALAR);
    attrHandle = H5Acreate1(groupHandle, "t_max", H5T_IEEE_F32LE, attrSpace, H5P_DEFAULT);
    H5Awrite(attrHandle, H5T_IEEE_F32LE, &t_max);
    H5Sclose(attrSpace);
    H5Aclose(attrHandle);

    switchHDF5ErrorHandling(true);
    H5Gclose(groupHandle);
}

void PLImg::HDF5Writer::write_dataset(const std::string& dataset, const cv::Mat& image) {
    hid_t dataHandle, spaceHandle;
    hid_t dtype, memType;
    switchHDF5ErrorHandling(false);
    dataHandle = H5Dopen(m_hdf5File, dataset.c_str(), H5P_DEFAULT);
    if(H5Eget_current_stack() >= 0) {
        hsize_t dims[2] = {static_cast<hsize_t>(image.cols), static_cast<hsize_t>(image.rows)};
        spaceHandle = H5Screate_simple(2, dims, nullptr);
        switch(image.type()) {
            case CV_32FC1:
                dtype = H5T_IEEE_F32LE;
                memType = H5T_NATIVE_FLOAT;
                break;
            case CV_32SC1:
                dtype = H5T_STD_I32LE;
                memType = H5T_NATIVE_INT;
                break;
            case CV_8UC1:
                dtype = H5T_STD_U8LE;
                memType = H5T_NATIVE_UCHAR;
                break;
            default:
                std::cout << "Datatype is currently not supported. Please contact the maintainer of the program!" << std::endl;
                exit(EXIT_FAILURE);
        }
        // This means that no dataset with this name is present
        dataHandle = H5Dcreate(m_hdf5File, dataset.c_str(), dtype, spaceHandle, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataHandle, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, image.data);

        H5Sclose(spaceHandle);
        H5Dclose(dataHandle);
    } else {
        // This means that a dataset is already present. We'll try to override the data if the image dimensions are the same.
    }
    switchHDF5ErrorHandling(true);
}

void PLImg::HDF5Writer::create_group(const std::string& group) {
    std::stringstream ss(group);
    std::string token;
    std::string groupString;
    herr_t status;
    switchHDF5ErrorHandling(false);
    while (std::getline(ss, token, '/')) {
        groupString.append("/").append(token);
        if(!token.empty()) {
            status = H5Gget_objinfo(m_hdf5File, groupString.c_str(), 0, nullptr);
            if (status != 0) {
                if(H5Gcreate1(m_hdf5File, groupString.c_str(), 0) != 0) {
                    H5Eprint2(H5Eget_current_stack(), stderr);
                    std::cerr << "Could not create group " << groupString << "! Exiting..." << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    switchHDF5ErrorHandling(true);

}

void PLImg::HDF5Writer::close() {
    switchHDF5ErrorHandling(true);
    H5Fclose(m_hdf5File);
}

void PLImg::HDF5Writer::open() {
    createDirectoriesIfMissing(m_filename);
    if(PLImg::reader::fileExists(m_filename)) {
        m_hdf5File = H5Fopen(m_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    } else {
        m_hdf5File = H5Fcreate(m_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }
    // first we need to create the parent group
    herr_t status = H5Gget_objinfo(m_hdf5File, "/", 0, nullptr);
    if(status != 0) {
        switchHDF5ErrorHandling(false);
        if (H5Gcreate1(m_hdf5File, "/", 0) != 0) {
            H5Eprint2(H5Eget_current_stack(), stderr);
            std::cerr << "Could not create group " << "/" << "! Exiting..." << std::endl;
            exit(EXIT_FAILURE);
        }
        switchHDF5ErrorHandling(true);
    }
}

void PLImg::HDF5Writer::createDirectoriesIfMissing(const std::string &filename) {
    // Get folder name
    auto pos = filename.find_last_of('/');
    if(pos != std::string::npos) {
        std::string folder_name = filename.substr(0, filename.find_last_of('/'));
        std::error_code err;
        std::filesystem::create_directory(folder_name, err);
        if(err.value() != 0) {
            throw std::runtime_error("Output folder " + folder_name + " could not LE created! Please check your path and permissions");
        }
    }
}

void PLImg::HDF5Writer::switchHDF5ErrorHandling(bool on) {
    if(on) {
        H5Eset_auto(H5E_DEFAULT, errorFunction, errorFunctionData);
    } else {
        /* Save old error handler */
        H5Eget_auto(H5E_DEFAULT, &errorFunction, &errorFunctionData);
        /* Turn off error handling */
        H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
    }
}


