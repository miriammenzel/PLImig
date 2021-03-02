#include <iostream>
#include <CLI/CLI.hpp>
#include "maskgeneration.h"
#include "reader.h"

int main(int argc, char** argv) {
    CLI::App app;

    std::vector<std::string> transmittance_files;
    std::vector<std::string> retardation_files;
    std::string output_folder;
    std::string dataset;
    int num_iterations;
    int num_retakes;
    int scale_factor;

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
    optional->add_option("--nit", num_iterations, "Number of iterations for the blurred mask")
            ->default_val(500);
    optional->add_option("--retakes", num_retakes, "Number of retakes")
            ->default_val(100);
    optional->add_option("--scaleFactor", scale_factor, "Scale subimages in blurring algorithm by 1/n")
            ->default_val(10);
    CLI11_PARSE(app, argc, argv);

    PLImg::MaskGeneration generation;
    std::string transmittance_basename, retardation_basename, param_basename;
    std::string retardation_path;
    bool retardation_found;
    for(const auto& transmittance_path : transmittance_files) {
        unsigned long long int endPosition = transmittance_path.find_last_of('/');
        if (endPosition != std::string::npos) {
            transmittance_basename = transmittance_path.substr(endPosition + 1);
        } else {
            transmittance_basename = transmittance_path;
        }
        for (const std::string& extension : std::array<std::string, 5>{".h5", ".tiff", ".tif", ".nii.gz", ".nii"}) {
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
            std::shared_ptr<cv::Mat> transmittance = std::make_shared<cv::Mat>(
                    PLImg::Reader::imread(transmittance_path, dataset));
            std::shared_ptr<cv::Mat> retardation = std::make_shared<cv::Mat>(
                    PLImg::Reader::imread(retardation_path, dataset));
            std::cout << "Files read" << std::endl;

            std::shared_ptr<cv::Mat> medTransmittance = transmittance;
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

            std::cout << "Will run " << num_retakes << " takes with " << num_iterations << " each" << std::endl;
            for (int take = 0; take < num_retakes; ++take) {
                std::ofstream param_file;
                param_basename = std::string(retardation_basename);
                if (param_basename.find("Retardation") != std::string::npos) {
                    param_basename = param_basename.replace(param_basename.find("Retardation"), 11, "Param_" + std::to_string(take));
                }
                param_file.open(output_folder + "/" + param_basename + ".csv");
                //////////////////////////////
                //////////////////////////////
                /// BLURRED MASK ALGORITHM ///
                //////////////////////////////
                //////////////////////////////

                std::shared_ptr<cv::Mat> small_retardation = std::make_shared<cv::Mat>(retardation->rows / scale_factor,
                                                                                       retardation->cols / scale_factor,
                                                                                       CV_32FC1);
                std::shared_ptr<cv::Mat> small_transmittance = std::make_shared<cv::Mat>(medTransmittance->rows / scale_factor,
                                                                                         medTransmittance->cols / scale_factor,
                                                                                         CV_32FC1);
                PLImg::MaskGeneration blurredGeneration(small_retardation, small_transmittance);
                int numPixels = retardation->rows * retardation->cols;

                uint num_threads;
                #pragma omp parallel default(shared)
                num_threads = omp_get_num_threads();

                std::vector<std::mt19937> random_engines(num_threads);
                #pragma omp parallel for default(shared) schedule(static)
                for (unsigned i = 0; i < num_threads; ++i) {
                    random_engines.at(i) = std::mt19937((clock() * i) % LONG_MAX);
                }
                std::uniform_int_distribution<int> distribution(0, numPixels);
                int selected_element;

                std::vector<float> above_tRet;
                std::vector<float> below_tRet;
                std::vector<float> above_tTra;
                std::vector<float> below_tTra;

                float t_ret, t_tra;
                float diff_tRet_p, diff_tRet_m, diff_tTra_p, diff_tTra_m;
                for (unsigned i = 0; i < num_iterations; ++i) {
                    std::cout << "\rBlurred Mask Generation: Iteration " << i << " of " << num_iterations;
                    std::flush(std::cout);
                    // Fill transmittance and retardation with random pixels from our base images
                    #pragma omp parallel for firstprivate(distribution, selected_element) schedule(static) default(shared)
                    for (int y = 0; y < small_retardation->rows; ++y) {
                        for (int x = 0; x < small_retardation->cols; ++x) {
                            selected_element = distribution(random_engines.at(omp_get_thread_num()));
                            small_retardation->at<float>(y, x) = retardation->at<float>(
                                    selected_element / retardation->cols, selected_element % retardation->cols);
                            small_transmittance->at<float>(y, x) = medTransmittance->at<float>(
                                    selected_element / medTransmittance->cols, selected_element % medTransmittance->cols);
                        }
                    }

                    blurredGeneration.setModalities(small_retardation, small_transmittance);
                    blurredGeneration.set_tMin(generation.tMin());
                    blurredGeneration.set_tMax(generation.tMax());

                    t_ret = blurredGeneration.tRet();
                    if (t_ret > generation.tRet()) {
                        above_tRet.push_back(t_ret);
                    } else if (t_ret < generation.tRet()) {
                        below_tRet.push_back(t_ret);
                    }

                    t_tra = blurredGeneration.tTra();
                    if (t_tra > generation.tTra()) {
                        above_tTra.push_back(t_tra);
                    } else if (t_tra < generation.tTra() && t_tra > 0) {
                        below_tTra.push_back(t_tra);
                    }

                    if (above_tRet.empty()) {
                        diff_tRet_p = generation.tRet();
                    } else {
                        diff_tRet_p = std::accumulate(above_tRet.begin(), above_tRet.end(), 0.0f) / above_tRet.size();
                    }
                    if (below_tRet.empty()) {
                        diff_tRet_m = generation.tRet();
                    } else {
                        diff_tRet_m = std::accumulate(below_tRet.begin(), below_tRet.end(), 0.0f) / below_tRet.size();
                    }
                    if (above_tTra.empty()) {
                        diff_tTra_p = generation.tTra();
                    } else {
                        diff_tTra_p = std::accumulate(above_tTra.begin(), above_tTra.end(), 0.0f) / above_tTra.size();
                    }
                    if (below_tTra.empty()) {
                        diff_tTra_m = generation.tTra();
                    } else {
                        diff_tTra_m = std::accumulate(below_tTra.begin(), below_tTra.end(), 0.0f) / below_tTra.size();
                    }

                    param_file << i << "," << diff_tRet_p << "," << generation.tRet() << "," << diff_tRet_m << ","
                                           << diff_tTra_p << "," << generation.tTra() << "," << diff_tTra_m << std::endl;
                }
                std::cout << std::endl;
                param_file.flush();
                param_file.close();
            }
        }
    }
}