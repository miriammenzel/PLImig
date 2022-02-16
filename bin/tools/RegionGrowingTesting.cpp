//
// Created by jreuter on 11.02.21.
//

#include <iostream>
#include <CLI/CLI.hpp>
#include "maskgeneration.h"
#include "reader.h"

int main(int argc, char** argv) {
    CLI::App app;

    std::vector <std::string> transmittance_files;
    std::vector <std::string> retardation_files;
    std::string output_folder;
    std::string dataset;
    float minPercent;
    float maxPercent;
    float stepPercent;
    bool generateInclination;

    auto required = app.add_option_group("Required parameters");
    required->add_option("--itra", transmittance_files, "Input transmittance files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("--iret", retardation_files, "Input retardation files")
            ->required()
            ->check(CLI::ExistingFile);
    required->add_option("-o, --output", output_folder, "Output folder")
            ->required()
            ->check(CLI::ExistingDirectory);

    auto optional = app.add_option_group("Optional parameters");
    optional->add_option("-d, --dataset", dataset, "HDF5 dataset")
            ->default_val("/Image");
    optional->add_option("--minPercent", minPercent)
            ->default_val(0.001);
    optional->add_option("--maxPercent", maxPercent)
            ->default_val(0.05);
    optional->add_option("--stepPercent", stepPercent)
            ->default_val(0.001);
    optional->add_flag("--withInclination", generateInclination);
    CLI11_PARSE(app, argc, argv);

    PLImg::MaskGeneration generation;
    std::string transmittance_basename, retardation_basename, param_basename;
    std::string retardation_path;
    bool retardation_found;
    for (const auto &transmittance_path : transmittance_files) {
        unsigned long long int endPosition = transmittance_path.find_last_of('/');
        if (endPosition != std::string::npos) {
            transmittance_basename = transmittance_path.substr(endPosition + 1);
        } else {
            transmittance_basename = transmittance_path;
        }
        for (const std::string &extension : std::array < std::string, 5 > {".h5", ".tiff", ".tif", ".nii.gz", ".nii"}) {
            endPosition = transmittance_basename.rfind(extension);
            if (endPosition != std::string::npos) {
                transmittance_basename = transmittance_basename.substr(0, endPosition);
            }
        }

        // Get name of retardation and check if transmittance has median filer applied.
        retardation_basename = std::string(transmittance_basename);
        if (retardation_basename.find("median10") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("median10"), 8, "");
        }
        if (retardation_basename.find("NTransmittance") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("NTransmittance"), 14,
                                                                "Retardation");
        }
        if (retardation_basename.find("Transmittance") != std::string::npos) {
            retardation_basename = retardation_basename.replace(retardation_basename.find("Transmittance"), 13,
                                                                "Retardation");
        }
        retardation_found = false;
        for (auto &retardation_file : retardation_files) {
            if (retardation_file.find(retardation_basename) != std::string::npos) {
                retardation_found = true;
                retardation_path = retardation_file;
                break;
            }
        }

        if (retardation_found) {
            std::shared_ptr <cv::Mat> transmittance = std::make_shared<cv::Mat>(
                    PLImg::Reader::imread(transmittance_path, dataset));
            std::shared_ptr <cv::Mat> retardation = std::make_shared<cv::Mat>(
                    PLImg::Reader::imread(retardation_path, dataset));
            std::cout << "Files read" << std::endl;

            std::shared_ptr <cv::Mat> medTransmittance = transmittance;
            if (transmittance_path.find("median10") == std::string::npos) {
                // Generate med10Transmittance
                medTransmittance = PLImg::cuda::filters::medianFilter(transmittance);
                // Write it to a file
                std::string medTraName(retardation_basename);
                medTraName.replace(retardation_basename.find("Retardation"), 10, "median10NTransmittance");
            } else {
                medTransmittance = transmittance;
            }
            transmittance = nullptr;
            std::cout << "Med10Transmittance generated" << std::endl;
            generation.setModalities(retardation, medTransmittance);

            float tMin;
            uint numberOfMaskPixels;
            std::ofstream param_file;

            param_basename = std::string(retardation_basename);
            if (param_basename.find("Retardation") != std::string::npos) {
                param_basename = param_basename.replace(param_basename.find("Retardation"), 11, "Param");
            }
            param_file.open(output_folder + "/" + param_basename + ".csv");
            param_file << "it,tMin,tTra,NPixels" << std::endl;

            for(float it = minPercent; it < maxPercent; it += stepPercent) {
                std::cout << "Iteration: " << it << " / " << maxPercent << std::endl;
                std::flush(std::cout);

                cv::Mat mask = PLImg::cuda::labeling::largestAreaConnectedComponents(*retardation, cv::Mat(), it);
                cv::Scalar mean = cv::mean(*medTransmittance, mask);
                tMin = mean[0];
                numberOfMaskPixels = cv::countNonZero(mask);
                generation.resetParameters();
                generation.set_tref(tMin);

                param_file << it << "," << tMin << "," << generation.T_thres() << "," << numberOfMaskPixels << std::endl;
                param_file.flush();
            }
            std::cout << std::endl;
            param_file.close();
        }
    }
}