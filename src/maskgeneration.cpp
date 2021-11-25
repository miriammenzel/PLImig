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
        m_retardation(std::move(retardation)), m_transmittance(std::move(transmittance)), m_tMin(nullptr), m_tMax(nullptr),
        m_tRet(nullptr), m_tTra(nullptr), m_whiteMask(nullptr), m_grayMask(nullptr), m_probabilityMask(nullptr) {
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
    this->m_tMin = nullptr;
    this->m_tMax = nullptr;
    this->m_tRet = nullptr;
    this->m_tTra = nullptr;
    this->m_whiteMask = nullptr;
    this->m_grayMask = nullptr;
    this->m_fullMask = nullptr;
    this->m_probabilityMask = nullptr;
}

void PLImg::MaskGeneration::removeBackground() {
    auto transmittanceThreshold = this->tMax();
    m_transmittance->setTo(m_maxTransmittance, *m_transmittance > transmittanceThreshold);
    m_retardation->setTo(m_minRetardation, *m_transmittance > transmittanceThreshold);
}

void PLImg::MaskGeneration::set_tMax(float tMax) {
    this->m_tMax = std::make_unique<float>(tMax);
}

void PLImg::MaskGeneration::set_tMin(float tMin) {
    this->m_tMin = std::make_unique<float>(tMin);
}

void PLImg::MaskGeneration::set_tRet(float tRet) {
    this->m_tRet = std::make_unique<float>(tRet);
}

void PLImg::MaskGeneration::set_tTra(float tTra) {
    this->m_tTra = std::make_unique<float>(tTra);
}

float PLImg::MaskGeneration::tTra() {
    if(!m_tTra) {
        float temp_tTra = tMin();

        // Generate histogram for potential correction of tMin for tTra
        cv::Mat hist = PLImg::cuda::histogram(*m_transmittance, m_minTransmittance, m_maxTransmittance, MAX_NUMBER_OF_BINS);

        int startPosition = temp_tTra / (float(m_maxTransmittance) - float(m_minTransmittance)) * float(MAX_NUMBER_OF_BINS);
        int endPosition = tMax() / (float(m_maxTransmittance) - float(m_minTransmittance)) * float(MAX_NUMBER_OF_BINS);

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
                this->m_tTra = std::make_unique<float>(m_minTransmittance + startPosition * stepSize);
            } else {
                this->m_tTra = std::make_unique<float>(m_minTransmittance + kappaPeaks.at(0) * stepSize);
            }
        } else {
            this->m_tTra = std::make_unique<float>(temp_tTra);
        }
    }
    return *this->m_tTra;
}

float PLImg::MaskGeneration::tRet() {
    if(!m_tRet) {
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
        this->m_tRet = std::make_unique<float>(temp_tRet);
    }
    return *this->m_tRet;
}

float PLImg::MaskGeneration::tMin() {
    if(!m_tMin) {
        cv::Mat backgroundMask = *m_retardation > 0 & *m_transmittance > 0 & *m_transmittance < tMax();
        cv::Mat mask = cuda::labeling::largestAreaConnectedComponents(*m_retardation, backgroundMask);
        cv::Scalar mean = cv::mean(*m_transmittance, mask);
        m_tMin = std::make_unique<float>(mean[0]);
    }
    return *this->m_tMin;
}

float PLImg::MaskGeneration::tMax() {
    if(!m_tMax) {
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
            this->m_tMax = std::make_unique<float>(m_minTransmittance + startPosition * stepSize);
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
            this->m_tMax = std::make_unique<float>(temp_tMax);
        }
    }
    return *this->m_tMax;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::grayMask() {
    if(!m_grayMask) {
        cv::Mat mask = (*m_transmittance >= tTra()) & (*m_transmittance <= tMax()) & (*m_retardation <= tRet());
        m_grayMask = std::make_shared<cv::Mat>(mask);
    }
    return m_grayMask;
}

std::shared_ptr<cv::Mat> PLImg::MaskGeneration::whiteMask() {
    if(!m_whiteMask) {
        cv::Mat mask = ((*m_transmittance < tTra()) & (*m_transmittance > 0)) | (*m_retardation > tRet());
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
        std::vector<float> above_tRet;
        std::vector<float> below_tRet;
        std::vector<float> above_tTra;
        std::vector<float> below_tTra;
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
        #pragma omp parallel shared(numberOfThreads, above_tRet, above_tTra, below_tRet, below_tTra, numberOfFinishedIterations)
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

                float t_ret, t_tra;
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
                    generation.set_tMin(this->tMin());
                    generation.set_tMax(this->tMax());

                    t_ret = generation.tRet();
                    if (t_ret >= this->tRet()) {
                        #pragma omp critical
                        above_tRet.push_back(t_ret);
                    } else if (t_ret <= this->tRet()) {
                        #pragma omp critical
                        below_tRet.push_back(t_ret);
                    }

                    t_tra = generation.tTra();
                    if (t_tra >= this->tTra()) {
                        #pragma omp critical
                        above_tTra.push_back(t_tra);
                    } else if (t_tra <= this->tTra() && t_tra > 0) {
                        #pragma omp critical
                        below_tTra.push_back(t_tra);
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
        if (above_tRet.empty()) {
            diff_tRet_p = tRet();
        } else {
            diff_tRet_p = std::accumulate(above_tRet.begin(), above_tRet.end(), 0.0f) / above_tRet.size();
        }
        if (below_tRet.empty()) {
            diff_tRet_m = tRet();
        } else {
            diff_tRet_m = std::accumulate(below_tRet.begin(), below_tRet.end(), 0.0f) / below_tRet.size();
        }
        if (above_tTra.empty()) {
            diff_tTra_p = tTra();
        } else {
            diff_tTra_p = std::accumulate(above_tTra.begin(), above_tTra.end(), 0.0f) / above_tTra.size();
        }
        if (below_tTra.empty()) {
            diff_tTra_m = tTra();
        } else {
            diff_tTra_m = std::accumulate(below_tTra.begin(), below_tTra.end(), 0.0f) / below_tTra.size();
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
            if(diffTra < tTra()) {
                diffTra = (diffTra - tTra()) / diff_tTra_m;
            } else {
                diffTra = (diffTra - tTra()) / diff_tTra_p;
            }
            diffRet = retardationPtr[idx];
            if(diffRet < tRet()) {
                diffRet = (diffRet - tRet()) / diff_tRet_m;
            } else {
                diffRet = (diffRet - tRet()) / diff_tRet_p;
            }
            probabilityMaskPtr[idx] =
                (-erf(cos(3.0f * M_PI / 4.0f - atan2f(diffTra, diffRet)) *
                sqrtf(diffTra * diffTra + diffRet * diffRet) * 2) + 1) / 2.0f;
        }
    }
    return m_probabilityMask;
}

