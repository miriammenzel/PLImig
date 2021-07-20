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

#ifdef TIME_MEASUREMENT
    #pragma message("Time measurement enabled.")
    #include <chrono>
#endif

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

    std::cout << "Transmittance range: " << m_minTransmittance << " -- " << m_maxTransmittance << "\n"
              << "Retardation range: " << m_minRetardation << " -- " << m_maxRetardation << std::endl;
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
    this->m_probabilityMask = nullptr;
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
        cv::Mat backgroundMask =  *m_retardation > 0 & *m_transmittance > 0 & *m_transmittance < tMax();
        cv::Mat mask = Image::largestAreaConnectedComponents(*m_retardation, cv::Mat());
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
        endPosition = std::max_element(fullHist.begin<float>() + MAX_NUMBER_OF_BINS / 2, fullHist.end<float>()) - fullHist.begin<float>();

        float histMaximum = endPosition * (m_maxTransmittance - m_minTransmittance) / MAX_NUMBER_OF_BINS + m_minTransmittance;
        fullHist = PLImg::cuda::histogram(*m_transmittance, m_minTransmittance,
                                          histMaximum,
                                          MAX_NUMBER_OF_BINS);
        fullHist.convertTo(fullHist, CV_32FC1);
        endPosition = MAX_NUMBER_OF_BINS - 1;

        auto peaks = PLImg::Histogram::peaks(fullHist, MAX_NUMBER_OF_BINS / 2, endPosition);
        if(!peaks.empty()) {
            startPosition = std::min_element(fullHist.begin<float>() + peaks.at(peaks.size() - 1),
                                             fullHist.begin<float>() + endPosition) - fullHist.begin<float>();
        } else {
            int width = Histogram::peakWidth(fullHist, endPosition, -1);
            startPosition = endPosition - 10 * width;
        }

        //If the transmittance was masked, we should see a large maxCurvature with 0 values after the highest peak
        if(endPosition - startPosition < 2) {
            this->m_tMax = std::make_unique<float>(float(startPosition)/MAX_NUMBER_OF_BINS);
        }
        //Else do the normal calculation
        else {
            // Convert from 256 to 64 bins
            endPosition = MIN_NUMBER_OF_BINS;
            startPosition = fmin(endPosition - 1, MIN_NUMBER_OF_BINS * float(startPosition)/MAX_NUMBER_OF_BINS);

            float temp_tMax;
            for(unsigned NUMBER_OF_BINS = MIN_NUMBER_OF_BINS; NUMBER_OF_BINS <= MAX_NUMBER_OF_BINS; NUMBER_OF_BINS = NUMBER_OF_BINS << 1) {
                cv::Mat hist(NUMBER_OF_BINS, 1, CV_32FC1);
                for (unsigned i = 0; i < NUMBER_OF_BINS; ++i) {
                    unsigned myStartPos = i * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                    unsigned myEndPos = (i + 1) * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                    hist.at<float>(i) = std::accumulate(fullHist.begin<float>() + myStartPos,
                                                        fullHist.begin<float>() + myEndPos, 0.0f);
                }
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
    cv::Mat mask = *whiteMask() | *grayMask();
    return std::make_shared<cv::Mat>(mask);
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

        #pragma omp parallel 
        {
            std::shared_ptr<cv::Mat> small_retardation = std::make_shared<cv::Mat>(m_retardation->rows/2, m_retardation->cols/2, CV_32FC1);
            std::shared_ptr<cv::Mat> small_transmittance = std::make_shared<cv::Mat>(m_transmittance->rows/2, m_transmittance->cols/2, CV_32FC1);
            MaskGeneration generation(small_retardation, small_transmittance);
            unsigned long long numPixels = (unsigned long long) m_retardation->rows *  (unsigned long long) m_retardation->cols;

            std::mt19937 random_engine = std::mt19937((clock() * omp_get_thread_num()) % LONG_MAX);
            std::uniform_int_distribution<unsigned long long> distribution(0, numPixels);
            unsigned long long selected_element;
            float t_ret, t_tra;

            for(unsigned i = 0; i < PROBABILITY_MASK_ITERATIONS / omp_get_num_threads(); ++i) {
                std::cout << "\rProbability Mask Generation: Iteration " << i << " of " << PROBABILITY_MASK_ITERATIONS / omp_get_num_threads() << std::endl;
                //std::flush(std::cout);         
                // Fill transmittance and retardation with random pixels from our base images
                for(int y = 0; y < small_retardation->rows; ++y) {
                    for (int x = 0; x < small_retardation->cols; ++x) {
                        selected_element = distribution(random_engine);
                        small_retardation->at<float>(y, x) = m_retardation->at<float>(
                                int(selected_element / m_retardation->cols), int(selected_element % m_retardation->cols));
                        small_transmittance->at<float>(y, x) = m_transmittance->at<float>(
                                int(selected_element / m_transmittance->cols), int(selected_element % m_transmittance->cols));
                    }
                }

                generation.setModalities(small_retardation, small_transmittance);
                generation.set_tMin(this->tMin());
                generation.set_tMax(this->tMax());

                t_ret = generation.tRet();
                if(t_ret >= this->tRet()) {
                    #pragma omp critical
                    above_tRet.push_back(t_ret);
                } else if(t_ret <= this->tRet()) {
                    #pragma omp critical
                    below_tRet.push_back(t_ret);
                }

                t_tra = generation.tTra();
                if(t_tra >= this->tTra()) {
                    #pragma omp critical
                    above_tTra.push_back(t_tra);
                } else if(t_tra <= this->tTra() && t_tra > 0) {
                    #pragma omp critical
                    below_tTra.push_back(t_tra);
                }
            }
            small_transmittance = nullptr;
            small_retardation = nullptr;
            generation.setModalities(nullptr, nullptr);
        }

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
        #pragma omp parallel for private(diffTra, diffRet) default(shared) schedule(static)
        for(int y = 0; y < m_retardation->rows; ++y) {
            for (int x = 0; x < m_retardation->cols; ++x) {
                diffTra = m_transmittance->at<float>(y, x);
                if(diffTra < tTra()) {
                    diffTra = (diffTra - tTra()) / diff_tTra_m;
                } else {
                    diffTra = (diffTra - tTra()) / diff_tTra_p;
                }
                diffRet = m_retardation->at<float>(y, x);
                if(diffRet < tRet()) {
                    diffRet = (diffRet - tRet()) / diff_tRet_m;
                } else {
                    diffRet = (diffRet - tRet()) / diff_tRet_p;
                }
                m_probabilityMask->at<float>(y, x) = (-erf(cos(3.0f * M_PI / 4.0f - atan2f(diffTra, diffRet)) *
                                                            sqrtf(diffTra * diffTra + diffRet * diffRet) * 2) + 1) / 2.0f;
            }
        }
    }
    return m_probabilityMask;
}

