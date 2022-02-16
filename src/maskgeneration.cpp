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

#include "maskgeneration.h"

PLImg::MaskGeneration::MaskGeneration(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance) :
        m_retardation(std::move(retardation)), m_transmittance(std::move(transmittance)), m_tref(nullptr), m_tback(nullptr),
        m_rthres(nullptr), m_tthres(nullptr), m_whiteMask(nullptr), m_grayMask(nullptr), m_probabilityMask(nullptr) {
    if(m_transmittance) {
        cv::minMaxIdx(*m_transmittance, &m_minTransmittance, &m_maxTransmittance);
        m_minTransmittance = fmax(m_minTransmittance, 0.0f);
    } else {
        m_minTransmittance = 0;
        m_maxTransmittance = 1;
    }
    if(m_retardation) {
        cv::minMaxIdx(*m_retardation, &m_minRetardation, &m_maxRetardation);
        m_minRetardation = fmax(m_minRetardation, 0.0f);
    } else {
        m_minRetardation = 0;
        m_maxRetardation = 1;
    }
}

void PLImg::MaskGeneration::setModalities(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance) {
    this->m_retardation = std::move(retardation);
    this->m_transmittance = std::move(transmittance);
    resetParameters();

    if(m_transmittance) {
        cv::minMaxIdx(*m_transmittance, &m_minTransmittance, &m_maxTransmittance);
        m_minTransmittance = fmax(m_minTransmittance, 0.0f);
    } else {
        m_minTransmittance = 0;
        m_maxTransmittance = 1;
    }
    if(m_retardation) {
        cv::minMaxIdx(*m_retardation, &m_minRetardation, &m_maxRetardation);
        m_minRetardation = fmax(m_minRetardation, 0.0f);
    } else {
        m_minRetardation = 0;
        m_maxRetardation = 1;
    }
}

void PLImg::MaskGeneration::resetParameters() {
    this->m_tref = nullptr;
    this->m_tback = nullptr;
    this->m_rthres = nullptr;
    this->m_tthres = nullptr;
    this->m_whiteMask = nullptr;
    this->m_grayMask = nullptr;
    this->m_fullMask = nullptr;
    this->m_probabilityMask = nullptr;
}

void PLImg::MaskGeneration::removeBackground() {
    auto transmittanceThreshold = this->T_back();
    m_transmittance->setTo(m_maxTransmittance, *m_transmittance > transmittanceThreshold);
    m_retardation->setTo(m_minRetardation, *m_transmittance > transmittanceThreshold);
}

void PLImg::MaskGeneration::set_tback(float tMax) {
    this->m_tback = std::make_unique<float>(tMax);
}

void PLImg::MaskGeneration::set_tref(float tMin) {
    this->m_tref = std::make_unique<float>(tMin);
}

void PLImg::MaskGeneration::set_rthres(float tRet) {
    this->m_rthres = std::make_unique<float>(tRet);
}

void PLImg::MaskGeneration::set_tthres(float tTra) {
    this->m_tthres = std::make_unique<float>(tTra);
}

float PLImg::MaskGeneration::T_thres() {
    if(!m_tthres) {
        float temp_tTra = T_ref();

        // Generate histogram for potential correction of tMin for tTra
        cv::Mat hist = PLImg::cuda::histogram(*m_transmittance, m_minTransmittance, m_maxTransmittance, MAX_NUMBER_OF_BINS);

        int startPosition = temp_tTra / (float(m_maxTransmittance) - float(m_minTransmittance)) * float(MAX_NUMBER_OF_BINS);
        int endPosition = T_back() / (float(m_maxTransmittance) - float(m_minTransmittance)) * float(MAX_NUMBER_OF_BINS);

        if(startPosition > endPosition) {
            int tmp = startPosition;
            startPosition = endPosition;
            endPosition = tmp;
        }

        auto peaks = Histogram::peaks(hist, startPosition, endPosition);

        if(peaks.size() > 0) {
            endPosition = std::min_element(hist.begin<float>() + startPosition, hist.begin<float>() + peaks.at(0)) - hist.begin<float>();
            float stepSize = (m_maxTransmittance - m_minTransmittance) / MAX_NUMBER_OF_BINS;
            auto kappa = Histogram::curvature(hist, m_minTransmittance, m_maxTransmittance);
            auto kappaPeaks = PLImg::Histogram::peaks(kappa, startPosition, endPosition);
            if(kappaPeaks.empty()) {
                this->m_tthres = std::make_unique<float>(m_minTransmittance + startPosition * stepSize);
            } else {
                this->m_tthres = std::make_unique<float>(m_minTransmittance + kappaPeaks.at(0) * stepSize);
            }
        } else {
            this->m_tthres = std::make_unique<float>(temp_tTra);
        }
    }
    return *this->m_tthres;
}

float PLImg::MaskGeneration::R_thres() {
    if(!m_rthres) {
        cv::Mat intHist = PLImg::cuda::histogram(*m_retardation, m_minRetardation + 1e-15, m_maxRetardation, MAX_NUMBER_OF_BINS);
        cv::Mat hist;
        intHist.convertTo(hist, CV_32FC1);
        cv::normalize(hist, hist, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);

        float temp_tRet = 0;
        int histogramMinimalBin = 0;
        float histogramMinimalValue = m_minRetardation;

        auto peaks = PLImg::Histogram::peaks(hist, 0, MAX_NUMBER_OF_BINS / 2);
        int startPosition, endPosition;
        startPosition = 0;
        if(!peaks.empty()) {
            histogramMinimalBin = peaks.at(peaks.size() - 1);
            histogramMinimalValue = histogramMinimalBin * (m_maxRetardation - m_minRetardation)/MAX_NUMBER_OF_BINS + m_minRetardation;
        }
        hist(cv::Range(histogramMinimalBin, MAX_NUMBER_OF_BINS), cv::Range(0, 1)).copyTo(hist);

        int width = Histogram::peakWidth(hist, startPosition, 1);
        endPosition = ceil(MIN_NUMBER_OF_BINS * 20.0f * width / MAX_NUMBER_OF_BINS);

        for(unsigned NUMBER_OF_BINS = MIN_NUMBER_OF_BINS; NUMBER_OF_BINS <= MAX_NUMBER_OF_BINS; NUMBER_OF_BINS *= 2) {
            hist = PLImg::cuda::histogram(*m_retardation, histogramMinimalValue, m_maxRetardation, NUMBER_OF_BINS);
            cv::normalize(hist, hist, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);

            auto kappa = Histogram::curvature(hist, histogramMinimalValue, m_maxRetardation);
            cv::normalize(kappa, kappa, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC1);

            // If more than one prominent peak is in the histogram, start at the second peak and not at the beginning
            auto peaks = PLImg::Histogram::peaks(hist, startPosition, endPosition);
            if(!peaks.empty()) {
                startPosition = peaks.at(peaks.size() - 1);
            }

            int resultingBin;
            auto kappaPeaks = PLImg::Histogram::peaks(kappa, startPosition, endPosition);
            if(kappaPeaks.empty()) {
                resultingBin = std::max_element(kappa.begin<float>() + startPosition, kappa.begin<float>() + endPosition) - kappa.begin<float>();
            } else {
                resultingBin = kappaPeaks.at(0);
            }

            float stepSize = float(m_maxRetardation - histogramMinimalValue) / float(NUMBER_OF_BINS);
            temp_tRet = histogramMinimalValue + float(resultingBin) * stepSize;
            // If our next step would be still in bounds for our histogram.
            startPosition = fmax(0, (resultingBin - 2) * 2 - 1);
            endPosition = fmin((resultingBin + 2) * 2 + 1, NUMBER_OF_BINS << 1);
        }
        this->m_rthres = std::make_unique<float>(temp_tRet);
    }
    return *this->m_rthres;
}

float PLImg::MaskGeneration::T_ref() {
    if(!m_tref) {
        cv::Mat backgroundMask = *m_retardation > 0 & *m_transmittance > 0 & *m_transmittance < T_back();
        cv::Mat mask = cuda::labeling::largestAreaConnectedComponents(*m_retardation, backgroundMask);
        cv::Scalar mean = cv::mean(*m_transmittance, mask);
        m_tref = std::make_unique<float>(mean[0]);
    }
    return *this->m_tref;
}

float PLImg::MaskGeneration::T_back() {
    if(!m_tback) {
        cv::Mat fullHist = PLImg::cuda::histogram(*m_transmittance, m_minTransmittance, m_maxTransmittance, MAX_NUMBER_OF_BINS);
        fullHist.convertTo(fullHist, CV_32FC1);

        // Determine start and end on full histogram
        int startPosition, endPosition;
        startPosition = MAX_NUMBER_OF_BINS / 3;
        endPosition = std::max_element(fullHist.begin<float>() + startPosition, fullHist.end<float>()) - fullHist.begin<float>();
        float histMaximum = endPosition * (m_maxTransmittance - m_minTransmittance) / MAX_NUMBER_OF_BINS + m_minTransmittance;
        fullHist = PLImg::cuda::histogram(*m_transmittance, m_minTransmittance,
                                          histMaximum,
                                          MAX_NUMBER_OF_BINS);
        fullHist.convertTo(fullHist, CV_32FC1);
        endPosition = MAX_NUMBER_OF_BINS - 1;

        auto peaks = PLImg::Histogram::peaks(fullHist, MAX_NUMBER_OF_BINS / 2, endPosition-1);
        if(!peaks.empty()) {
            startPosition = std::min_element(fullHist.begin<float>() + peaks.at(peaks.size() - 1),
                                             fullHist.begin<float>() + endPosition) - fullHist.begin<float>();
        } else {
            int width = Histogram::peakWidth(fullHist, endPosition, -1);
            startPosition = endPosition - 10 * width;
        }

        //If the transmittance was masked, we should see a large maxCurvature with 0 values after the highest peak
        if(endPosition - startPosition < 2) {
            float stepSize = float(histMaximum - m_minTransmittance) / float(MAX_NUMBER_OF_BINS);
            this->m_tback = std::make_unique<float>(m_minTransmittance + startPosition * stepSize);
        }
        //Else do the normal calculation
        else {
            // Convert from 256 to 64 bins
            endPosition = MIN_NUMBER_OF_BINS;
            startPosition = fmin(endPosition - 1, MIN_NUMBER_OF_BINS * float(startPosition)/MAX_NUMBER_OF_BINS);

            float temp_tMax;
            for(unsigned NUMBER_OF_BINS = MIN_NUMBER_OF_BINS; NUMBER_OF_BINS <= MAX_NUMBER_OF_BINS; NUMBER_OF_BINS = NUMBER_OF_BINS << 1) {
                cv::Mat hist = PLImg::cuda::histogram(*m_transmittance, m_minTransmittance,
                                                      histMaximum,
                                                      NUMBER_OF_BINS);
                cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, CV_32FC1);

                float stepSize = float(histMaximum - m_minTransmittance) / float(NUMBER_OF_BINS);
                auto kappa = Histogram::curvature(hist, m_minTransmittance, histMaximum);
                auto kappaPeaks = Histogram::peaks(kappa, startPosition+1, endPosition-1);
                int resultingBin;

                if (kappaPeaks.empty()) {
                    resultingBin = std::max_element(kappa.begin<float>() + startPosition,
                                                    kappa.begin<float>() + endPosition) - kappa.begin<float>();
                } else {
                    resultingBin = kappaPeaks.at(kappaPeaks.size() - 1);
                }

                temp_tMax = m_minTransmittance + resultingBin * stepSize;
                startPosition = fmax(0, (resultingBin * 2 - 1));
                endPosition = NUMBER_OF_BINS << 1;
            }
            this->m_tback = std::make_unique<float>(temp_tMax);
        }
    }
    return *this->m_tback;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::grayMask() {
    if(!m_grayMask) {
        cv::Mat mask = (*m_transmittance >= T_thres()) & (*m_transmittance <= T_back()) & (*m_retardation <= R_thres());
        m_grayMask = std::make_shared<cv::Mat>(mask);
    }
    return m_grayMask;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::whiteMask() {
    if(!m_whiteMask) {
        cv::Mat mask = ((*m_transmittance < T_thres()) & (*m_transmittance > 0)) | (*m_retardation > R_thres());
        m_whiteMask = std::make_shared<cv::Mat>(mask);
    }
    return m_whiteMask;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::fullMask() {
    if(!m_fullMask) {
        if (!m_whiteMask) whiteMask();
        if (!m_grayMask) grayMask();

        cv::Mat mask(m_whiteMask->rows, m_whiteMask->cols, CV_8UC1);
        mask.setTo(0);
        mask.setTo(200, *whiteMask());
        mask.setTo(100, *grayMask());
        m_fullMask = std::make_shared<cv::Mat>(mask);
    }
    return m_fullMask;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::noNerveFiberMask() {
    cv::Mat backgroundMask;
    cv::Scalar mean, stddev;
    cv::bitwise_not(*fullMask(), backgroundMask);
    cv::meanStdDev(*m_retardation, mean, stddev, backgroundMask);
    cv::Mat mask = *m_retardation < mean[0] + 2*stddev[0] & *grayMask();
    return std::make_shared<cv::Mat>(mask);
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::probabilityMask() {
    if(!m_probabilityMask) {
        std::vector<float> above_rthres;
        std::vector<float> below_rthres;
        std::vector<float> above_tthres;
        std::vector<float> below_tthres;
        m_probabilityMask = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_32FC1);

        // We're trying to calculate the maximum possible number of threads than can be used simultaneously to calculate multiple iterations at once.
        float predictedMemoryUsage = PLImg::cuda::getHistogramMemoryEstimation(Image::randomizedModalities(m_transmittance, m_retardation, 0.5f)[0], MAX_NUMBER_OF_BINS);
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
        #pragma omp parallel shared(numberOfThreads, above_rthres, below_rthres, above_tthres, below_tthres, numberOfFinishedIterations)
        {
            #pragma omp single
            {
                std::cout << "Computing " << numberOfThreads << " iterations in parallel with max. " << omp_get_max_threads() / numberOfThreads << " threads per iteration." << std::endl;
            }
            #ifdef __GNUC__
                omp_set_num_threads(omp_get_max_threads() / numberOfThreads);
            #endif

            // Only work with valid threads. The other threads won't do any work.
            if(omp_get_thread_num() < numberOfThreads) {
                std::shared_ptr<cv::Mat> small_retardation;
                std::shared_ptr<cv::Mat> small_transmittance;
                MaskGeneration generation(small_retardation, small_transmittance);

                float r_thres, t_thres;
                unsigned int ownNumberOfIterations = PROBABILITY_MASK_ITERATIONS / numberOfThreads;
                int overhead = PROBABILITY_MASK_ITERATIONS % numberOfThreads;
                if (overhead > 0 && omp_get_thread_num() < overhead) {
                    ++ownNumberOfIterations;
                }

                for (unsigned int i = 0; i < ownNumberOfIterations; ++i) {
                    auto small_modalities = Image::randomizedModalities(m_transmittance, m_retardation, 0.5f);
                    small_transmittance = std::make_shared<cv::Mat>(small_modalities[0]);
                    small_retardation = std::make_shared<cv::Mat>(small_modalities[1]);

                    generation.setModalities(small_retardation, small_transmittance);
                    generation.set_tref(this->T_ref());
                    generation.set_tback(this->T_back());

                    r_thres = generation.R_thres();
                    if (r_thres >= this->R_thres()) {
                        #pragma omp critical
                        above_rthres.push_back(r_thres);
                    } else if (r_thres <= this->R_thres()) {
                        #pragma omp critical
                        below_rthres.push_back(r_thres);
                    }

                    t_thres = generation.T_thres();
                    if (t_thres >= this->T_thres()) {
                        #pragma omp critical
                        above_tthres.push_back(t_thres);
                    } else if (t_thres <= this->T_thres() && t_thres > 0) {
                        #pragma omp critical
                        below_tthres.push_back(t_thres);
                    }

                    #pragma omp critical
                    {
                        ++numberOfFinishedIterations;
                        std::cout << "\rProbability Mask Generation: Iteration " << numberOfFinishedIterations << " of "
                                  << PROBABILITY_MASK_ITERATIONS;
                        std::flush(std::cout);
                    };
                }
                small_transmittance = nullptr;
                small_retardation = nullptr;
                generation.setModalities(nullptr, nullptr);
            }
        }
        #ifdef __GNUC__
            omp_set_max_active_levels(omp_levels);
        #endif
        #if _OPENMP < 201611
            omp_set_nested(false);
        #endif

        std::cout << std::endl;

        float diff_tRet_p, diff_tRet_m, diff_tTra_p, diff_tTra_m;
        if (above_rthres.empty()) {
            diff_tRet_p = R_thres();
        } else {
            diff_tRet_p = std::accumulate(above_rthres.begin(), above_rthres.end(), 0.0f) / above_rthres.size();
        }
        if (below_rthres.empty()) {
            diff_tRet_m = R_thres();
        } else {
            diff_tRet_m = std::accumulate(below_rthres.begin(), below_rthres.end(), 0.0f) / below_rthres.size();
        }
        if (above_tthres.empty()) {
            diff_tTra_p = T_thres();
        } else {
            diff_tTra_p = std::accumulate(above_tthres.begin(), above_tthres.end(), 0.0f) / above_tthres.size();
        }
        if (below_tthres.empty()) {
            diff_tTra_m = T_thres();
        } else {
            diff_tTra_m = std::accumulate(below_tthres.begin(), below_tthres.end(), 0.0f) / below_tthres.size();
        }

        std::cout << "Probability parameters: R+:"  << diff_tRet_p << ", R-:" << diff_tRet_m <<
                                                    ", T+:" << diff_tTra_p << ", T-:" << diff_tTra_m
                                                    << std::endl;

        float diffTra = 0.0f;
        float diffRet = 0.0f;

        // Get pointers from OpenCV matrices to prevent overflow errors when image is larger than UINT_MAX
        float* probabilityMaskPtr = (float*) m_probabilityMask->data;
        const float* transmittancePtr = (float*) m_transmittance->data;
        const float* retardationPtr = (float*) m_retardation->data;

        // Calculate probability mask
        #pragma omp parallel for private(diffTra, diffRet) default(shared) schedule(static)
        for(unsigned long long idx = 0; idx < ((unsigned long long) m_probabilityMask->rows * m_probabilityMask->cols); ++idx) {
            diffTra = transmittancePtr[idx];
            if(diffTra < T_thres()) {
                diffTra = (diffTra - T_thres()) / diff_tTra_m;
            } else {
                diffTra = (diffTra - T_thres()) / diff_tTra_p;
            }
            diffRet = retardationPtr[idx];
            if(diffRet < R_thres()) {
                diffRet = (diffRet - R_thres()) / diff_tRet_m;
            } else {
                diffRet = (diffRet - R_thres()) / diff_tRet_p;
            }
            probabilityMaskPtr[idx] =
                (-erf(cos(3.0f * M_PI / 4.0f - atan2f(diffTra, diffRet)) *
                sqrtf(diffTra * diffTra + diffRet * diffRet) * 2) + 1) / 2.0f;
        }
    }
    return m_probabilityMask;
}

