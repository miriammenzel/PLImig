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

#include "CLI/CLI.hpp"

#include <fstream>
#include <H5Cpp.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app;

    std::vector<std::string> files;
    std::string output_folder;
    std::string dataset;
    auto required = app.add_option_group("Required parameters");
    required->add_option("-i, --input", files, "Input mask files from PLImg")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("-o, --output", output_folder, "Output folder")
            ->required()
            ->check(CLI::ExistingDirectory);

    auto optional = app.add_option_group("Optional parameters");
    optional->add_option("-d, --dataset", dataset, "HDF5 dataset")
            ->default_val("/Image");

    CLI11_PARSE(app, argc, argv);

    std::ofstream statusFile;
    statusFile.open(output_folder + "/erroneous_results.txt", std::ios::out | std::ios::app);

    // Disable HDF5 exceptions as they are pretty annoying in the terminal
    H5::Exception::dontPrint();

    std::array<std::string, 4> maskAttributes = {"r_thres", "I_lower", "I_rmax", "I_upper"};
    std::array<std::string, 4> inclinationAttributes = {"im", "ic", "rmax_G", "rmax_W"};
    for(const auto& file : files) {
        H5::H5File input;
        input.openFile(file, H5F_ACC_RDONLY);

        bool valuesValid = true;

        H5::Attribute attr;
        float attrValue;

        // Check for mask attributes
        for(const std::string& attrName : maskAttributes) {
            if (input.attrExists(attrName)) {
                attr = input.openAttribute(attrName);
                attr.read(H5::PredType::NATIVE_FLOAT, &attrValue);
                attr.close();

                if(std::isnan(attrValue) || std::isinf(attrValue) || attrValue < 0) {
                    valuesValid = false;
                }
            }
        }

        try {
            H5::Group group = input.openGroup(dataset);
            // Check for inclination attributes
            for(const std::string& attrName : inclinationAttributes) {
                if (group.attrExists(attrName)) {
                    attr = group.openAttribute(attrName);
                    attr.read(H5::PredType::NATIVE_FLOAT, &attrValue);
                    attr.close();
                    if(std::isnan(attrValue) || std::isinf(attrValue) || attrValue < 0) {
                        valuesValid = false;
                    }
                }
            }
        } catch(H5::FileIException& exception) {
            try {
                H5::DataSet dset = input.openDataSet(dataset);
                // Check for inclination attributes
                for(const std::string& attrName : inclinationAttributes) {
                    if (dset.attrExists(attrName)) {
                        attr = dset.openAttribute(attrName);
                        attr.read(H5::PredType::NATIVE_FLOAT, &attrValue);
                        attr.close();
                        if(std::isnan(attrValue) || std::isinf(attrValue)) {
                            valuesValid = false;
                        }
                    }
                }
            } catch(H5::FileIException& exception) {}
        }

        if(!valuesValid) {
            statusFile << file << std::endl;
        }

        input.close();
    }

    return EXIT_SUCCESS;
}