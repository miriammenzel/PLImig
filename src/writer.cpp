//
// Created by jreuter on 25.11.20.
//

#include "writer.h"
#include <iostream>

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
    while(dataset.size() > 0 && dataset.at(dataset.size()-1) == '/') {
        dataset = dataset.substr(0, dataset.size()-1);
    }
    if(m_hdf5File->atexists(dataset+"t_tra")) {
        m_hdf5File->atdelete(dataset+"t_tra");
    }
    m_hdf5File->atwrite(t_tra, dataset+"t_tra");

    if(m_hdf5File->atexists(dataset+"t_ret")) {
        m_hdf5File->atdelete(dataset+"t_ret");
    }
    m_hdf5File->atwrite(t_ret, dataset+"t_ret");

    if(m_hdf5File->atexists(dataset+"t_min")) {
        m_hdf5File->atdelete(dataset+"t_min");
    }
    m_hdf5File->atwrite(t_min, dataset+"t_min");

    if(m_hdf5File->atexists(dataset+"t_max")) {
        m_hdf5File->atdelete(dataset+"t_max");
    }
    m_hdf5File->atwrite(t_max, dataset+"t_max");
}

void PLImg::HDF5Writer::write_dataset(const std::string& dataset, const cv::Mat& image) {
    if (!m_hdf5File->hlexists(dataset)) {
        m_hdf5File->dscreate(image.rows, image.cols, image.type(), dataset);
    }
    m_hdf5File->dswrite(image, dataset);
}

void PLImg::HDF5Writer::create_group(const std::string& group) {
    std::stringstream ss(group);
    std::string token;
    std::string groupString;
    while (std::getline(ss, token, '/')) {
        groupString.append("/").append(token);
        if(!token.empty()) {
            if (!m_hdf5File->hlexists(groupString)) {
                m_hdf5File->grcreate(groupString);
            }
        }
    }

}

void PLImg::HDF5Writer::close() {
    m_hdf5File->close();
    m_hdf5File.release();
}

void PLImg::HDF5Writer::open() {
    createDirectoriesIfMissing(m_filename);
    m_hdf5File = cv::hdf::open( m_filename );
    // first we need to create the parent group
    if (!m_hdf5File->hlexists("/")) {
        m_hdf5File->grcreate("/");
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
