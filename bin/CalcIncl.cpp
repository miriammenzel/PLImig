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

#include "reader.h"
#include "writer.h"
#include "inclination.h"
#include "CLI/CLI.hpp"
#include <opencv2/core.hpp>

#include <vector>
#include <string>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app;

    // Get the number of threads for all following steps
    int numThreads;
    #pragma omp parallel
    numThreads = omp_get_num_threads();
    cv::setNumThreads(numThreads);

    std::vector<std::string> transmittance_files;
    std::vector<std::string> retardation_files;
    std::vector<std::string> mask_files;
    std::string output_folder;
    std::string dataset;
    float im, ic, rmaxWhite, rmaxGray;
    bool detailed = false;

    auto required = app.add_option_group("Required parameters");
    required->add_option("--itra", transmittance_files, "Input transmittance files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("--iret", retardation_files, "Input retardation files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("--imask", mask_files, "Input mask files from PLImg")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("-o, --output", output_folder, "Output folder")
            ->required()
            ->check(CLI::ExistingDirectory);

    auto optional = app.add_option_group("Optional parameters");
    optional->add_option("-d, --dataset", dataset, "HDF5 dataset")
            ->default_val("/Image");
    optional->add_flag("--detailed", detailed);
    optional->add_option("--im", im)->default_val(-1);
    optional->add_option("--ic", ic)->default_val(-1);
    optional->add_option("--rmaxWhite", rmaxWhite)->default_val(-1);
    optional->add_option("--rmaxGray", rmaxGray)->default_val(-1);

    CLI11_PARSE(app, argc, argv);

    PLImg::HDF5Writer writer;
    PLImg::Inclination inclination;
    std::string transmittance_basename, retardation_basename, mask_basename, inclination_basename;
    std::string retardation_path, mask_path;
    bool retardation_found, mask_found;

    PLImg::cuda::runCUDAchecks();
    for(const auto& transmittance_path : transmittance_files) {
        std::cout << transmittance_path << std::endl;

        unsigned long long int endPosition = transmittance_path.find_last_of('/');
        if(endPosition != std::string::npos) {
            transmittance_basename = transmittance_path.substr(endPosition+1);
        } else {
            transmittance_basename = transmittance_path;
        }
        for(std::string extension : std::array<std::string, 5> {".h5", ".tiff", ".tif", ".nii.gz", ".nii"}) {
            endPosition = transmittance_basename.rfind(extension);
            if(endPosition != std::string::npos) {
                transmittance_basename = transmittance_basename.substr(0, endPosition);
            }
        }

        // Get name of retardation and check if transmittance has median filer applied.
        retardation_basename = std::string(transmittance_basename);
        if (retardation_basename.find("median10") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("median10"), 8, "");
        }
        if (retardation_basename.find("NTransmittance") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("NTransmittance"), 14, "Retardation");
        }
        if (retardation_basename.find("Transmittance") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("Transmittance"), 13, "Retardation");
        }
        retardation_found = false;
        for(auto & retardation_file : retardation_files) {
            if(retardation_file.find(retardation_basename) != std::string::npos) {
                retardation_found = true;
                retardation_path = retardation_file;
                break;
            }
        }

        mask_basename = std::string(retardation_basename);
        if (mask_basename.find("Retardation") != std::string::npos) {
            mask_basename = mask_basename.replace(mask_basename.find("Retardation"), 11, "Mask");
        }
        mask_found = false;
        for(auto & mask_file : mask_files) {
            if(mask_file.find(mask_basename) != std::string::npos) {
                mask_found = true;
                mask_path = mask_file;
                break;
            }
        }

        if (retardation_found && mask_found) {
            inclination_basename = std::string(mask_basename);
            if (mask_basename.find("Mask") != std::string::npos) {
                inclination_basename = inclination_basename.replace(inclination_basename.find("Mask"), 4, "Inclination");
            }

            // Read all files.
            std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(PLImg::Reader::imread(transmittance_path, dataset));
            std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(PLImg::Reader::imread(retardation_path, dataset));
            std::shared_ptr<cv::Mat> mask = std::make_shared<cv::Mat>(PLImg::Reader::imread(mask_path, dataset));
            std::shared_ptr<cv::Mat> blurredMask = std::make_shared<cv::Mat>(PLImg::Reader::imread(mask_path, "/Probability"));
            std::cout << "Files read" << std::endl;

            std::shared_ptr<cv::Mat> medTransmittance;
            // If our given transmittance isn't already median filtered (based on it's file name)
            if (transmittance_path.find("median10") == std::string::npos) {
                // Generate med10Transmittance
                medTransmittance = PLImg::cuda::filters::medianFilterMasked(transmittance, mask);
            } else {
                medTransmittance = transmittance;
            }
            transmittance = nullptr;
            std::cout << "Med10Transmittance generated" << std::endl;

            // Set our read parameters
            inclination.setModalities(medTransmittance, retardation, blurredMask, mask);
            // If manual parameters were given, apply them here
            if(im >= 0) {
                inclination.set_im(im);
            }
            if(ic >= 0) {
                inclination.set_ic(ic);
            }
            if(rmaxWhite >= 0) {
                inclination.set_rmaxWhite(rmaxWhite);
            }
            if(rmaxGray >= 0) {
                inclination.set_rmaxGray(rmaxGray);
            }
            // Create file and dataset. Write the inclination afterwards.
            writer.set_path(output_folder+ "/" + inclination_basename + ".h5");
            writer.write_dataset("/Image", *inclination.inclination(), true);
            writer.write_attribute("/Image", "im", inclination.im());
            writer.write_attribute("/Image", "ic", inclination.ic());
            writer.write_attribute("/Image", "rmax_W", inclination.rmaxWhite());
            writer.write_attribute("/Image", "rmax_G", inclination.rmaxGray());
            writer.write_attribute("/Image", "version", PLImg::Version::versionHash() + ", " + PLImg::Version::timeStamp());

            writer.writePLIMAttributes(transmittance_path, retardation_path, "/Image", "/Image", "Inclination", argc, argv);
            std::cout << "Inclination generated and written" << std::endl;
            writer.close();

            if(detailed) {
                auto saturation_basename = std::string(mask_basename);
                if (mask_basename.find("Mask") != std::string::npos) {
                    saturation_basename = saturation_basename.replace(saturation_basename.find("Mask"), 4, "Saturation");
                }
                // Create file and dataset. Write the inclination afterwards.
                writer.set_path(output_folder+ "/" + saturation_basename + ".h5");

                writer.write_dataset("/Image", *inclination.saturation(), true);
                writer.write_attribute("/Image", "im", inclination.im());
                writer.write_attribute("/Image", "ic", inclination.ic());
                writer.write_attribute("/Image", "rmax_W", inclination.rmaxWhite());
                writer.write_attribute("/Image", "rmax_G", inclination.rmaxGray());
                writer.write_attribute("/Image", "version", PLImg::Version::versionHash() + ", " + PLImg::Version::timeStamp());

                writer.writePLIMAttributes(transmittance_path, retardation_path, "/Image", "/Image", "Inclination Saturation", argc, argv);
                std::cout << "Saturation image generated and written" << std::endl;
                writer.close();
            }

            std::cout << std::endl;
        } else {
            std::cerr << "Mask or Retardation not found. Please check your paths!\n";
            std::cerr << "Retardation : " << retardation_path << "\n";
            std::cerr << "Mask : " << mask_path << "\n" << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
