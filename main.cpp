#include "reader.h"
#include "writer.h"
#include "maskgeneration.h"
#include "CLI/CLI.hpp"

#include <vector>
#include <string>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app;

    std::vector<std::string> input_files;
    std::string output_folder;
    std::string dataset;
    bool detailed = false;
    bool blurred = false;

    app.add_option("-i, --input, ifile", input_files, "Input transmittance files")
                    ->required()
                    ->check(CLI::ExistingFile);
    app.add_option("-o, --output, ofolder", output_folder, "Output folder")
                    ->required()
                    ->check(CLI::ExistingDirectory);
    app.add_option("-d, --dataset, dset", dataset, "HDF5 dataset")
                    ->default_val("/Image");
    app.add_flag("--detailed", detailed);
    app.add_flag("--with_blurred", blurred);

    CLI11_PARSE(app, argc, argv);

    PLImg::HDF5Writer writer;
    PLImg::MaskGeneration generation;

    PLImg::filters::runCUDAchecks();
    for(auto file : input_files) {
        std::cout << file << std::endl;

        // Get name of retardation and check if transmittance has median filer applied.
        auto retardation_path = file;
        if (retardation_path.find("median10") != std::string::npos) {
            retardation_path = file.replace(file.find("median10"), 8, "");
        }
        if (retardation_path.find("NTransmittance") != std::string::npos) {
            retardation_path = retardation_path.replace(retardation_path.find("NTransmittance"), 14, "Retardation");
        }
        if (retardation_path.find("Transmittance") != std::string::npos) {
            retardation_path = retardation_path.replace(retardation_path.find("Transmittance"), 13, "Retardation");
        }
        auto basename = retardation_path.substr(retardation_path.find_last_of('/')+1);
        if (basename.find("Retardation") != std::string::npos) {
            basename = basename.replace(basename.find("Retardation"), 11, "Mask");
            basename = basename.substr(0, basename.find_last_of('.'));
        }

        std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(PLImg::imread(file, dataset));
        std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(PLImg::imread(retardation_path, dataset));
        std::cout << "Files read" << std::endl;

        std::shared_ptr<cv::Mat> medTransmittance = transmittance;
        if (file.find("median10") == std::string::npos) {
            // Generate med10Transmittance
            medTransmittance = PLImg::filters::medianFilter(transmittance, 10);
            // Write it to a file
            std::string medTraName(basename);
            medTraName.replace(basename.find("Mask"), 4, "median10NTransmittance");
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

        generation.setModalities(retardation, medTransmittance);
        writer.set_path(output_folder+ "/" + basename + ".h5");
        writer.create_group(dataset);
        writer.write_attributes("/", generation.tTra(), generation.tRet(), generation.tMin(), generation.tMax());
        std::cout << "Attributes generated and written" << std::endl;
        writer.write_dataset(dataset+"/White", *generation.whiteMask());
        std::cout << "White mask generated and written" << std::endl;
        writer.write_dataset(dataset+"/Gray", *generation.grayMask());
        std::cout << "Gray mask generated and written" << std::endl;

        if (blurred) {
            writer.write_dataset(dataset+"/Blurred", *generation.blurredMask());
            std::cout << "Blurred mask generated and written" << std::endl;
        }
        if (detailed) {
            writer.write_dataset(dataset+"/Full", *generation.fullMask());
            writer.write_dataset(dataset+"/NoNerveFibers", *generation.noNerveFiberMask());
            std::cout << "Detailed masks generated and written" << std::endl;
        }
        writer.close();
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}