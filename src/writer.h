/*
    MIT License

    Copyright (c) 2020 Forschungszentrum Jülich / Jan André Reuter.

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

#ifndef PLIMG_WRITER_H
#define PLIMG_WRITER_H

#include <filesystem>
#include <H5Cpp.h>
#include <opencv2/core.hpp>

#include "plim/AttributeHandler.h"
#include "reader.h"
#include "version.h"

/**
 * @file
 * @brief PLImg::HDF5Writer class
 */
namespace PLImg {
    /**
     * @brief The HDF5Writer class
     */
    class HDF5Writer {
    public:
        /**
         * @brief HDF5Writer Default constructor for the HDF5Writer
         * Default constructor for the HDF5Writer. This class will be used to write HDF5 files.
         */
        HDF5Writer();
        /**
         * @brief path Get currently set path
         * @return Currently set path via the set_path(const std::string& filename) method
         */
        std::string path();
        /**
         * @brief set_path Set HDF5 path. If path exists, open the file
         * @param filename Path of the file which will be written
         * Set the desired path of the HDF5 file which will be written. The file will be created if it doesn't exist.
         * If the file already exists, open it in append mode.
         */
        void set_path(const std::string& filename);
        /**
         * @brief write_attributes
         * @param dataset
         * @param parameter_name
         * @param value
         */
        template<typename T>
        void write_attribute(std::string dataset, const std::string& parameter_name, T value);
        /**
         * @brief write_dataset
         * @param dataset
         * @param image
         */
        void write_dataset(const std::string& dataset, const cv::Mat& image);
        /**
         * @brief create_group
         * @param group
         */
        void create_group(const std::string& group);
        /**
         * @brief close Closes currently opened file.
         */
        void close();
        /**
         *
         * @param transmittance_path Original transmittance path used for the call of the program
         * @param retardation_path Original retardation path used for the call of the program
         * @param output_dataset Dataset which will be written to
         * @param input_dataset Dataset which will be used to copy most attributes
         * @param modality Name of the modality which will be written (NTransmittance, Retardation, ...)
         * @param argc Number of arguments when calling the program
         * @param argv Arguments when calling the program
         */
        void writePLIMAttributes(const std::string& transmittance_path, const std::string& retardation_path,
                                 const std::string& output_dataset, const std::string& input_dataset,
                                 const std::string& modality, int argc, char** argv);
    private:
        /**
         * @brief open
         */
        void open();
        /**
         * @brief createDirectoriesIfMissing
         * @param filename
         */
        static void createDirectoriesIfMissing(const std::string& filename);

        void write_type_attribute(std::string dataset, const std::string& parameter_name, const H5::AtomType& datatype, void* value);

        ///
        std::string m_filename;
        ///
        H5::H5File m_hdf5file;
    };
}

#endif //PLIMG_WRITER_H

template<> void PLImg::HDF5Writer::write_attribute<float>(std::string dataset, const std::string& parameter_name, float value);
template<> void PLImg::HDF5Writer::write_attribute<double>(std::string dataset, const std::string& parameter_name, double value);
template<> void PLImg::HDF5Writer::write_attribute<int>(std::string dataset, const std::string& parameter_name, int value);
template<> void PLImg::HDF5Writer::write_attribute<std::string>(std::string dataset, const std::string& parameter_name, std::string value);