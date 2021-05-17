/*
    MIT License

    Copyright (c) 2021 Forschungszentrum Jülich / Jan André Reuter.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
 */

#include "writer.h"
#include <iostream>

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

void PLImg::HDF5Writer::write_attribute(const std::string& dataset, const std::string& parameter_name, float value) {
    this->write_type_attribute(dataset, parameter_name, H5::PredType::NATIVE_FLOAT, &value);
}

void PLImg::HDF5Writer::write_attribute(const std::string& dataset, const std::string& parameter_name, double value) {
    this->write_type_attribute(dataset, parameter_name, H5::PredType::NATIVE_DOUBLE, &value);
}

void PLImg::HDF5Writer::write_attribute(const std::string& dataset, const std::string& parameter_name, int value) {
    this->write_type_attribute(dataset, parameter_name, H5::PredType::NATIVE_INT, &value);
}

void PLImg::HDF5Writer::write_attribute(const std::string& dataset, const std::string& parameter_name, std::string value) {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    this->write_type_attribute(dataset, parameter_name, str_type, &value);
}

void PLImg::HDF5Writer::write_type_attribute(const std::string& dataset, const std::string& parameter_name, const H5::AtomType& type, void* value) {
    hsize_t dims[1] = {1};
    H5::Attribute attr;
    H5::DataSpace space(1, dims);
    std::string path_appendix;
    try {
        H5::Group grp = m_hdf5file.openGroup(dataset);
        if (!grp.attrExists(parameter_name)) {
            attr = grp.createAttribute(parameter_name, type, space);
        } else {
            attr = grp.openAttribute(parameter_name);
        }
    } catch(H5::FileIException& exception) {
        try {
            H5::DataSet dset = m_hdf5file.openDataSet(dataset);
            if (!dset.attrExists(parameter_name)) {
                attr = dset.createAttribute(parameter_name, type, space);
            } else {
                attr = dset.openAttribute(parameter_name);
            }
        } catch(H5::FileIException& exception) {
            return;
        }
    }
    attr.write(type, value);
    attr.close();
}

void PLImg::HDF5Writer::write_dataset(const std::string& dataset, const cv::Mat& image, bool create_softlink) {
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
        H5::PredType dtype = H5::PredType::NATIVE_FLOAT;
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

        if(create_softlink) {
            try {
                m_hdf5file.createGroup("/pyramid");
            } catch (...) {}
            m_hdf5file.link(H5G_LINK_SOFT, dataset, "/pyramid/00");
        }

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

void PLImg::HDF5Writer::writePLIMAttributes(const std::string& transmittance_path, const std::string& retardation_path,
                                            const std::string& output_dataset, const std::string& input_dataset,
                                            const std::string& modality, const int argc, char** argv) {
    H5::Exception::dontPrint();
    hid_t id;
    H5::Group grp;
    H5::DataSet dset;
        try {
        grp = m_hdf5file.openGroup(output_dataset);
        id = grp.getId();
    } catch(H5::FileIException& exception) {
        dset = m_hdf5file.openDataSet(output_dataset);
        id = dset.getId();
    }

    try {
        plim::AttributeHandler outputHandler(id);
        if(outputHandler.doesAttributeExist("image_modality")) {
            outputHandler.deleteAttribute("image_modality");
        }
        outputHandler.setStringAttribute("image_modality", modality);

        std::string username;
        uid_t uid = geteuid();
        struct passwd *pw = getpwuid(uid);
        if (pw) {
            username = pw->pw_name;
        }
        if(outputHandler.doesAttributeExist("created_by")) {
            outputHandler.deleteAttribute("created_by");
        }
        outputHandler.setStringAttribute("created_by", username);

        if(outputHandler.doesAttributeExist("creation_time")) {
            outputHandler.deleteAttribute("creation_time");
        }
        outputHandler.setStringAttribute("creation_time", Version::timeStamp());

        if(outputHandler.doesAttributeExist("software")) {
            outputHandler.deleteAttribute("software");
        }
        outputHandler.setStringAttribute("software", argv[0]);

        if(outputHandler.doesAttributeExist("software_revision")) {
            outputHandler.deleteAttribute("software_revision");
        }
        outputHandler.setStringAttribute("software_revision", Version::versionHash());

        std::string software_parameters;
        for(unsigned i = 1; i < argc; ++i) {
            software_parameters += std::string(argv[i]) + " ";
        }
        if(outputHandler.doesAttributeExist("software_parameters")) {
            outputHandler.deleteAttribute("software_parameters");
        }
        outputHandler.setStringAttribute("software_parameters", software_parameters);


        H5::H5File transmittance;
        H5::DataSet tr_dset;
        H5::H5File retardation;
        H5::DataSet ret_dset;
        std::unique_ptr<plim::AttributeHandler> transmittance_handler = nullptr;
        std::unique_ptr<plim::AttributeHandler> retardation_handler = nullptr;
        bool h5_transmittance = false;
        bool h5_retardation = false;

        try {
            if (transmittance_path.find(".h5") != std::string::npos) {
                transmittance.openFile(transmittance_path, H5F_ACC_RDONLY);
                tr_dset = transmittance.openDataSet(input_dataset);
                transmittance_handler = std::make_unique<plim::AttributeHandler>(tr_dset.getId());
                h5_transmittance = true;
            }
            if (retardation_path.find(".h5") != std::string::npos) {
                retardation.openFile(retardation_path, H5F_ACC_RDONLY);
                ret_dset = retardation.openDataSet(input_dataset);
                retardation_handler = std::make_unique<plim::AttributeHandler>(ret_dset.getId());
                h5_retardation = true;
            }

            if (h5_retardation) {
                retardation_handler->copyAllAttributesTo(outputHandler, {});
            } else if (h5_transmittance) {
                transmittance_handler->copyAllAttributesTo(outputHandler, {});
            }

            if (h5_retardation & !h5_transmittance) {
                outputHandler.setReferenceModalityTo({*retardation_handler});
            } else if (h5_transmittance & !h5_retardation) {
                outputHandler.setReferenceModalityTo({*transmittance_handler});
            } else if (h5_transmittance && h5_retardation) {
                outputHandler.setReferenceModalityTo({*transmittance_handler, *retardation_handler});
            }

            outputHandler.addCreator();
            outputHandler.addId();
        } catch(...) {
            std::cerr << "Error during copying attributes with plim. Skipping..." << std::endl;
        }
        transmittance_handler = nullptr;
        retardation_handler = nullptr;

        if(tr_dset.getId() > 0) tr_dset.close();
        if(transmittance.getId() > 0) transmittance.close();

        if(ret_dset.getId() > 0) ret_dset.close();
        if(retardation.getId() > 0) retardation.close();
    } catch (...) {
        throw std::runtime_error("Output dataset was not valid!");
    }

    grp.close();
    dset.close();
}


