#include <iostream>
#include <CLI/CLI.hpp>
#include "maskgeneration.h"
#include "reader.h"
#include "writer.h"

int main(int argc, char** argv) {
    CLI::App app;

    std::vector<std::string> transmittance_files;
    std::vector<std::string> retardation_files;
    std::string output_folder;
    std::string dataset;
    int num_iterations;
    int num_retakes;
    float scale_factor;

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
            ->default_val(10);
    optional->add_option("--scaleFactor", scale_factor, "Scale subimages in blurring algorithm by 1/n")
            ->default_val(0.1f);
    CLI11_PARSE(app, argc, argv);

    PLImg::MaskGeneration generation;
    std::string transmittance_basename, retardation_basename, param_basename;
    std::string retardation_path;
    bool retardation_found;
    for(const auto& transmittance_path : transmittance_files) {
        #ifdef WIN32
                unsigned long long int endPosition = transmittance_path.find_last_of('\\');
        #else
                unsigned long long int endPosition = transmittance_path.find_last_of('/');
        #endif
        
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
            if (transmittance_path.find("median") == std::string::npos) {
                // Generate med10Transmittance
                medTransmittance = PLImg::cuda::filters::medianFilter(transmittance);
            } else {
                medTransmittance = transmittance;
            }
            transmittance = nullptr;
            std::cout << "Med10Transmittance generated" << std::endl;
            generation.setModalities(retardation, medTransmittance);
            generation.removeBackground();
            generation.T_back();
            generation.T_ref();
            generation.R_thres();
            generation.T_thres();

            std::cout << "Will run " << num_retakes << " takes with " << num_iterations << " each" << std::endl;
            for (int take = 0; take < num_retakes; ++take) {
                std::ofstream param_file;
                param_basename = std::string(retardation_basename);
                if (param_basename.find("Retardation") != std::string::npos) {
                    param_basename = param_basename.replace(param_basename.find("Retardation"), 11, "Param_" + std::to_string(take));
                }
                #ifdef WIN32
                    param_file.open(output_folder + "\\" + param_basename + ".csv");
                #else
                    param_file.open(output_folder + "/" + param_basename + ".csv");
                #endif      
                
                //////////////////////////////
                //////////////////////////////
                /// BLURRED MASK ALGORITHM ///
                //////////////////////////////
                //////////////////////////////

                std::vector<float> above_tRet;
                std::vector<float> below_tRet;
                std::vector<float> above_tTra;
                std::vector<float> below_tTra;
                auto probabilityMask = std::make_shared<cv::Mat>(retardation->rows, retardation->cols, CV_32FC1);

                // We're trying to calculate the maximum possible number of threads than can be used simultaneously to calculate multiple iterations at once.
                float predictedMemoryUsage = PLImg::cuda::getHistogramMemoryEstimation(PLImg::Image::randomizedModalities(medTransmittance, retardation, scale_factor)[0], MAX_NUMBER_OF_BINS);
                // Calculate the number of threads that will be used based on the free memory and the maximum number of threads
                int numberOfThreads;
                #pragma omp parallel
                numberOfThreads = omp_get_num_threads();
                numberOfThreads = fmax(1, fmin(numberOfThreads, uint(float(PLImg::cuda::getFreeMemory()) / predictedMemoryUsage)));

                std::cout << "OpenMP version used during compilation (doesn't have to match the executing OpenMP version): " << _OPENMP << std::endl;
                #if _OPENMP < 201611
                    omp_set_nested(true);
                #endif
                #ifdef __GNUC__
                    auto omp_levels = omp_get_max_active_levels();
                    omp_set_max_active_levels(3);
                #endif
                ushort numberOfFinishedIterations = 0;
                #pragma omp parallel shared(numberOfThreads, above_tRet, above_tTra, below_tRet, below_tTra, numberOfFinishedIterations)
                {
                    #pragma omp single
                    {
                        std::cout << "Computing " << numberOfThreads << " iterations in parallel with max. " << omp_get_max_threads() / numberOfThreads << " threads per iteration." << std::endl;
                    }
                    omp_set_num_threads(omp_get_max_threads() / numberOfThreads);

                    // Only work with valid threads. The other threads won't do any work.
                    if(omp_get_thread_num() < numberOfThreads) {
                        std::shared_ptr<cv::Mat> small_retardation;
                        std::shared_ptr<cv::Mat> small_transmittance;
                        PLImg::MaskGeneration iter_generation(small_retardation, small_transmittance);

                        float t_ret, t_tra;
                        uint ownNumberOfIterations = num_iterations / numberOfThreads;
                        uint overhead = num_iterations % numberOfThreads;
                        if (overhead > 0 && omp_get_thread_num() < overhead) {
                            ++ownNumberOfIterations;
                        }

                        for (int i = 0; i < ownNumberOfIterations; ++i) {
                            auto small_modalities = PLImg::Image::randomizedModalities(medTransmittance, retardation, scale_factor);
                            small_transmittance = std::make_shared<cv::Mat>(small_modalities[0]);
                            small_retardation = std::make_shared<cv::Mat>(small_modalities[1]);

                            iter_generation.setModalities(small_retardation, small_transmittance);
                            iter_generation.set_tref(generation.T_ref());
                            iter_generation.set_tback(generation.T_back());

                            t_ret = iter_generation.R_thres();
                            t_tra = iter_generation.T_thres();

                            #pragma omp critical
                            {
                                if (t_tra >= generation.T_thres()) {
                                    above_tTra.push_back(t_tra);
                                } else if (t_tra <= generation.T_thres() && t_tra > 0) {
                                    below_tTra.push_back(t_tra);
                                }
                                if (t_ret >= generation.R_thres()) {
                                    above_tRet.push_back(t_ret);
                                } else if (t_ret <= generation.R_thres()) {
                                    below_tRet.push_back(t_ret);
                                }

                                ++numberOfFinishedIterations;
                                std::cout << "\rProbability Mask Generation: Iteration " << numberOfFinishedIterations << " of "
                                          << num_iterations;
                                std::flush(std::cout);

                                float diff_tRet_p, diff_tRet_m, diff_tTra_p, diff_tTra_m;
                                if (above_tRet.empty()) {
                                    diff_tRet_p = generation.R_thres();
                                } else {
                                    diff_tRet_p = std::accumulate(above_tRet.begin(), above_tRet.end(), 0.0f) / above_tRet.size();
                                }
                                if (below_tRet.empty()) {
                                    diff_tRet_m = generation.R_thres();
                                } else {
                                    diff_tRet_m = std::accumulate(below_tRet.begin(), below_tRet.end(), 0.0f) / below_tRet.size();
                                }
                                if (above_tTra.empty()) {
                                    diff_tTra_p = generation.T_thres();
                                } else {
                                    diff_tTra_p = std::accumulate(above_tTra.begin(), above_tTra.end(), 0.0f) / above_tTra.size();
                                }
                                if (below_tTra.empty()) {
                                    diff_tTra_m = generation.T_thres();
                                } else {
                                    diff_tTra_m = std::accumulate(below_tTra.begin(), below_tTra.end(), 0.0f) / below_tTra.size();
                                }

                                param_file << numberOfFinishedIterations << "," << diff_tRet_p << "," << generation.R_thres() << "," << diff_tRet_m << "," << t_ret << ","
                                                                         << diff_tTra_p << "," << generation.T_thres() << "," << diff_tTra_m << "," << t_tra << std::endl;
                            }
                        }
                    }
                }
                #ifdef __GNUC__
                    omp_set_max_active_levels(omp_levels);
                #endif
                #if _OPENMP < 201611
                    omp_set_nested(false);
                #endif

                std::cout << std::endl;
                param_file.flush();
                param_file.close();
            }
        }
    }
}

