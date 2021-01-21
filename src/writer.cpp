//
// Created by jreuter on 25.11.20.
//

#include "writer.h"

PLImg::HDF5Writer::HDF5Writer() {
    m_filename = "";
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

    hsize_t dims[1] = {1};
    H5::Attribute attr;
    H5::DataSpace space(1, dims);
    if(!m_hdf5file.attrExists(dataset+"t_tra")) {
        attr = m_hdf5file.createAttribute(dataset + "t_tra", H5::PredType::NATIVE_FLOAT, space);
    } else {
        attr = m_hdf5file.openAttribute(dataset + "t_tra");
    }
    attr.write(H5::PredType::NATIVE_FLOAT, &t_tra);
    attr.close();

    if(!m_hdf5file.attrExists(dataset+"t_ret")) {
        attr = m_hdf5file.createAttribute(dataset + "t_ret", H5::PredType::NATIVE_FLOAT, space);
    } else {
        attr = m_hdf5file.openAttribute(dataset + "t_ret");
    }
    attr.write(H5::PredType::NATIVE_FLOAT, &t_ret);
    attr.close();

    if(!m_hdf5file.attrExists(dataset+"t_min")) {
        attr = m_hdf5file.createAttribute(dataset + "t_min", H5::PredType::NATIVE_FLOAT, space);
    } else {
        attr = m_hdf5file.openAttribute(dataset + "t_min");
    }
    attr.write(H5::PredType::NATIVE_FLOAT, &t_min);
    attr.close();

    if(!m_hdf5file.attrExists(dataset+"t_max")) {
        attr = m_hdf5file.createAttribute(dataset + "t_max", H5::PredType::NATIVE_FLOAT, space);
    } else {
        attr = m_hdf5file.openAttribute(dataset + "t_max");
    }
    attr.write(H5::PredType::NATIVE_FLOAT, &t_max);
    attr.close();
}

void PLImg::HDF5Writer::write_dataset(const std::string& dataset, const cv::Mat& image) {
    H5::DataSet dset;
    H5::DataSpace dataSpace;
    hsize_t dims[2];
    H5::Exception::dontPrint();
    // Try to open the dataset.
    // This will throw an exception if the dataset doesn't exist.
    bool dataSetFound;
    try {
        dset = m_hdf5file.openDataSet(dataset);
        dataSetFound = true;
    } catch (...) {
        dataSetFound = false;
    }
    if(dataSetFound) {
        // If the dataset is found, the program cannot delete the existing dataset
        // Instead we will try to override the existing content if rows and columns do match
        dataSpace = dset.getSpace();
        dataSpace.getSimpleExtentDims(dims);
        if(dims[0] == image.rows && dims[1] == image.cols) {
            dset.write(image.data, dset.getDataType(), dataSpace);
        } else {
            throw std::runtime_error("Selected path is not empty and colums or rows do not match. Please check your path!");
        }
        dataSpace.close();
        dset.close();
    } else {
        // Create dataset normally
        // Check for the datatype from the OpenCV mat to determine the HDF5 datatype
        H5::DataType dtype;
        switch(image.type()) {
            case CV_32FC1:
                dtype = H5::PredType::NATIVE_FLOAT;
                break;
            case CV_32SC1:
                dtype = H5::PredType::NATIVE_INT;
                break;
            case CV_8UC1:
                dtype = H5::PredType::NATIVE_UINT8;
                break;
        }
        // Write dataset
        dims[0] = static_cast<hsize_t>(image.rows);
        dims[1] = static_cast<hsize_t>(image.cols);
        dataSpace = H5::DataSpace(2, dims);
        dset = m_hdf5file.createDataSet(dataset, dtype, dataSpace);
        dset.write(image.data, dtype);

        dset.close();
        dataSpace.close();
        dtype.close();
    }
}

void PLImg::HDF5Writer::create_group(const std::string& group) {
    std::stringstream ss(group);
    std::string token;
    std::string groupString;

    H5::Exception::dontPrint();
    H5::Group gr;
    // Create groups recursively if the group doesn't exist.
    while (std::getline(ss, token, '/')) {
        groupString.append("/").append(token);
        if(!token.empty()) {
            try {
                gr = m_hdf5file.createGroup(groupString);
                gr.close();
            } catch(...){}
        }
    }
}

void PLImg::HDF5Writer::close() {
    m_hdf5file.close();
}

void PLImg::HDF5Writer::open() {
    createDirectoriesIfMissing(m_filename);
    // If the file doesn't exist open it with Read-Write.
    // Otherwise open it with appending so that existing content will not be deleted.
    if(PLImg::Reader::fileExists(m_filename)) {
        try {
            m_hdf5file = H5::H5File(m_filename, H5F_ACC_RDWR);
        }  catch (...) {
            H5::Exception::printErrorStack();
            exit(EXIT_FAILURE);
        }
    } else {
        m_hdf5file = H5::H5File(m_filename, H5F_ACC_TRUNC);
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
            throw std::runtime_error("Output folder " + folder_name + " could not be created! Please check your path and permissions");
        }
    }
}


