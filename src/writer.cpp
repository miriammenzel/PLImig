//
// Created by jreuter on 25.11.20.
//

#include "writer.h"
#include <iostream>

void PLImg::HDF5Writer::set_path(const std::string& filename) {
    if(this->m_filename != filename) {
        this->m_filename = filename;
        this->open();
    }
}

void PLImg::HDF5Writer::write_attributes(const std::string& dataset, float t_tra, float t_ret, float t_min, float t_max) {
    /*
        image_dataset.attrs['created_by'] = getpass.getuser()
        image_dataset.attrs['software'] = sys.argv[0]
        image_dataset.attrs['software_parameters'] = ' '.join(sys.argv[1:])
        image_dataset.attrs['image_modality'] = 'Mask'
        image_dataset.attrs['filename'] = self.path
        image_dataset.attrs['t_tra'] = t_tra
        image_dataset.attrs['t_ret'] = t_ret
        image_dataset.attrs['tra_min'] = t_min
        image_dataset.attrs['tra_max'] = t_max
     */
    if(m_hdf5File->atexists(dataset+"/t_tra")) {
        m_hdf5File->atdelete(dataset+"/t_tra");
    }
    m_hdf5File->atwrite(t_tra, dataset+"/t_tra");

    if(m_hdf5File->atexists(dataset+"/t_ret")) {
        m_hdf5File->atdelete(dataset+"/t_ret");
    }
    m_hdf5File->atwrite(t_ret, dataset+"/t_ret");

    if(m_hdf5File->atexists(dataset+"/t_min")) {
        m_hdf5File->atdelete(dataset+"/t_min");
    }
    m_hdf5File->atwrite(t_min, dataset+"/t_min");

    if(m_hdf5File->atexists(dataset+"/t_max")) {
        m_hdf5File->atdelete(dataset+"/t_max");
    }
    m_hdf5File->atwrite(t_max, dataset+"/t_max");
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
    m_hdf5File = cv::hdf::open( m_filename );
    // first we need to create the parent group
    if (!m_hdf5File->hlexists("/")) {
        m_hdf5File->grcreate("/");
    }
}
