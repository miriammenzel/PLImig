#include "reader.h"
#include "writer.h"
#include "inclination.h"
#include "CLI/CLI.hpp"

#include <vector>
#include <string>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app;

    std::vector<std::string> transmittance_files;
    std::vector<std::string> retardation_files;
    std::vector<std::string> mask_files;
    std::string output_folder;
    std::string dataset;

    app.add_option("--itra, itra", transmittance_files, "Input transmittance files")
            ->required()
            ->check(CLI::ExistingFile);
    app.add_option("--iret, iret", retardation_files, "Input retardation files")
            ->required()
            ->check(CLI::ExistingFile);
    app.add_option("--imask, imask", mask_files, "Input mask files from PLImg")
            ->required()
            ->check(CLI::ExistingFile);
    app.add_option("-o, --output, ofolder", output_folder, "Output folder")
            ->required()
            ->check(CLI::ExistingDirectory);
    app.add_option("-d, --dataset, dset", dataset, "HDF5 dataset")
            ->default_val("/Image");

    CLI11_PARSE(app, argc, argv);

    PLImg::HDF5Writer writer;
    PLImg::Inclination inclination;
    std::string transmittance_basename, retardation_basename, mask_basename, inclination_basename;
    std::string retardation_path, mask_path;
    bool retardation_found, mask_found;

    PLImg::filters::runCUDAchecks();
    for(const auto& transmittance_path : transmittance_files) {
        std::cout << transmittance_path << std::endl;

        transmittance_basename = transmittance_path.substr(transmittance_path.find_last_of('/')+1);
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

            std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(PLImg::reader::imread(transmittance_path, dataset));
            std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(PLImg::reader::imread(retardation_path, dataset));
            std::shared_ptr<cv::Mat> whiteMask = std::make_shared<cv::Mat>(PLImg::reader::imread(mask_path, dataset+"/White"));
            std::shared_ptr<cv::Mat> grayMask = std::make_shared<cv::Mat>(PLImg::reader::imread(mask_path, dataset+"/Gray"));;
            std::shared_ptr<cv::Mat> blurredMask = std::make_shared<cv::Mat>(PLImg::reader::imread(mask_path, dataset+"/Blurred"));;
            std::cout << "Files read" << std::endl;

            std::shared_ptr<cv::Mat> medTransmittance;
            if (transmittance_path.find("median10") == std::string::npos) {
                // Generate med10Transmittance
                medTransmittance = PLImg::filters::medianFilter(transmittance, 10);
                // Write it to a file
                std::string medTraName(mask_basename);
                std::cout << mask_basename << std::endl;
                medTraName.replace(medTraName.find("Mask"), 4, "median10NTransmittance");
                // Set file
                writer.set_path(output_folder+ "/" + medTraName + ".h5");
                // Set dataset
                std::string group = dataset.substr(0, dataset.find_last_of('/'));
                // Create group and dataset
                writer.create_group(group);
                writer.write_dataset(dataset+"/", *medTransmittance);
                writer.close();
            } else {
                medTransmittance = transmittance;
            }
            transmittance = nullptr;
            std::cout << "Med10Transmittance generated" << std::endl;

            inclination.setModalities(medTransmittance, retardation, blurredMask, whiteMask, grayMask);
            writer.set_path(output_folder+ "/" + inclination_basename + ".h5");
            writer.create_group(dataset);
            writer.write_dataset(dataset+"/Inclination", *inclination.inclination());
            std::cout << "Inclination generated and written" << std::endl;

            writer.close();
            std::cout << std::endl;
        }
    }

    return EXIT_SUCCESS;
}