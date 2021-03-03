//
// Created by jreuter on 25.11.20.
//

#include "maskgeneration.h"

PLImg::MaskGeneration::MaskGeneration(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance) :
        m_retardation(std::move(retardation)), m_transmittance(std::move(transmittance)), m_tMin(nullptr), m_tMax(nullptr),
        m_tRet(nullptr), m_tTra(nullptr), m_whiteMask(nullptr), m_grayMask(nullptr), m_probabilityMask(nullptr) {
}

void PLImg::MaskGeneration::setModalities(std::shared_ptr<cv::Mat> retardation, std::shared_ptr<cv::Mat> transmittance) {
    this->m_retardation = std::move(retardation);
    this->m_transmittance = std::move(transmittance);
    resetParameters();
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
        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f+1e-15f};
        const float* histRange = { histBounds };
        int histSize = MAX_NUMBER_OF_BINS;

        cv::Mat hist;
        cv::calcHist(&(*m_transmittance), 1, channels, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

        int startPosition = temp_tTra * MAX_NUMBER_OF_BINS;
        int endPosition = tMax() * MAX_NUMBER_OF_BINS;

        auto peaks = Histogram::peaks(hist, startPosition, endPosition);
        if(peaks.size() > 0) {
            endPosition = std::min_element(hist.begin<float>() + startPosition, hist.begin<float>() + peaks.at(0)) - hist.begin<float>();
            this->m_tTra = std::make_unique<float>(
                    Histogram::maxCurvature(hist, 0.0f, 1.0f, 1, startPosition, endPosition));
        } else {
            this->m_tTra = std::make_unique<float>(temp_tTra);
        }
    }
    return *this->m_tTra;
}

float PLImg::MaskGeneration::tRet() {
    if(!m_tRet) {
        float temp_tRet = 0;
        int channels[] = {0};
        float histBounds[] = {0.0f+1e-10f, 1.0f};
        const float* histRange = { histBounds };

        int startPosition, endPosition;
        startPosition = 0;
        endPosition = MIN_NUMBER_OF_BINS / 2;

        // Generate histogram
        cv::Mat fullHist;
        int histSize = MAX_NUMBER_OF_BINS;
        cv::calcHist(&(*m_retardation), 1, channels, cv::Mat(), fullHist, 1, &histSize, &histRange, true, false);

        for(unsigned NUMBER_OF_BINS = MIN_NUMBER_OF_BINS; NUMBER_OF_BINS <= MAX_NUMBER_OF_BINS; NUMBER_OF_BINS = NUMBER_OF_BINS << 1) {
            cv::Mat hist(NUMBER_OF_BINS, 1, CV_32FC1);
            #pragma omp parallel for
            for(unsigned i = 0; i < NUMBER_OF_BINS; ++i) {
                unsigned myStartPos = i * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                unsigned myEndPos = (i+1) * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                hist.at<float>(i) = std::accumulate(fullHist.begin<float>() + myStartPos, fullHist.begin<float>() + myEndPos, 0);
            }
            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, CV_32F);

            // If more than one prominent peak is in the histogram, start at the second peak and not at the beginning
            auto peaks = PLImg::Histogram::peaks(hist, startPosition, endPosition, 1e-2f);
            if(peaks.size() > 1) {
                startPosition = peaks.at(peaks.size() - 1);
            } else if(peaks.size() == 1) {
                startPosition = peaks.at(0);
            }

            temp_tRet = Histogram::maxCurvature(hist, 0.0f, 1.0f, 1, startPosition, endPosition);

            startPosition = fmax(0, (temp_tRet * NUMBER_OF_BINS - 2) * ((NUMBER_OF_BINS << 1) / NUMBER_OF_BINS) - 1);
            endPosition = fmin((temp_tRet * NUMBER_OF_BINS + 2) * ((NUMBER_OF_BINS << 1) / NUMBER_OF_BINS) + 1, NUMBER_OF_BINS << 1);
        }

        this->m_tRet = std::make_unique<float>(temp_tRet);
    }
    return *this->m_tRet;
}

float PLImg::MaskGeneration::tMin() {
    if(!m_tMin) {
        cv::Mat backgroundMask = *m_transmittance > 0 & *m_transmittance < tMax() & *m_retardation > 0;
        cv::Mat mask = Image::regionGrowing(*m_retardation, backgroundMask);

        cv::Scalar mean = cv::mean(*m_transmittance, mask);
        m_tMin = std::make_unique<float>(mean[0]);
    }
    return *this->m_tMin;
}

float PLImg::MaskGeneration::tMax() {
    if(!m_tMax) {
        int channels[] = {0};
        float histBounds[] = {0.0f, 1.0f + 1.0f/MAX_NUMBER_OF_BINS};
        const float* histRange = { histBounds };
        int histSize = MAX_NUMBER_OF_BINS;

        cv::Mat fullHist;
        cv::calcHist(&(*m_transmittance), 1, channels, cv::Mat(), fullHist, 1, &histSize, &histRange, true, false);

        // Determine start and end on full histogram
        int startPosition, endPosition;
        endPosition = std::max_element(fullHist.begin<float>() + MAX_NUMBER_OF_BINS / 2, fullHist.end<float>()) - fullHist.begin<float>();
        auto peaks = PLImg::Histogram::peaks(fullHist, MAX_NUMBER_OF_BINS / 2, endPosition);
        if(peaks.size() > 1) {
            startPosition = std::min_element(fullHist.begin<float>() + peaks.at(peaks.size() - 1),
                                             fullHist.begin<float>() + endPosition) - fullHist.begin<float>();
        } else {
            startPosition = MAX_NUMBER_OF_BINS / 2;
        }

        //If the transmittance was masked, we should see a large maxCurvature with 0 values after the highest peak
        if(endPosition - startPosition < 2) {
            this->m_tMax = std::make_unique<float>(float(startPosition)/MAX_NUMBER_OF_BINS);
        }
        //Else do the normal calculation
        else {
            // Convert from 256 to 16 bins
            startPosition = MIN_NUMBER_OF_BINS * float(startPosition)/MAX_NUMBER_OF_BINS;
            endPosition = MIN_NUMBER_OF_BINS * float(endPosition)/MAX_NUMBER_OF_BINS;

            float temp_tMax;
            for(unsigned NUMBER_OF_BINS = MIN_NUMBER_OF_BINS; NUMBER_OF_BINS <= MAX_NUMBER_OF_BINS; NUMBER_OF_BINS = NUMBER_OF_BINS << 1) {
                cv::Mat hist(NUMBER_OF_BINS, 1, CV_32FC1);
                #pragma omp parallel for
                for (unsigned i = 0; i < NUMBER_OF_BINS; ++i) {
                    unsigned myStartPos = i * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                    unsigned myEndPos = (i + 1) * MAX_NUMBER_OF_BINS / NUMBER_OF_BINS;
                    hist.at<float>(i) = std::accumulate(fullHist.begin<float>() + myStartPos,
                                                        fullHist.begin<float>() + myEndPos, 0);
                }
                cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

                temp_tMax = Histogram::maxCurvature(hist, 0.0f, 1.0f, -1, startPosition, endPosition);

                startPosition = fmax(0, (temp_tMax * NUMBER_OF_BINS - 2) * ((NUMBER_OF_BINS << 1) / NUMBER_OF_BINS) - 1);
                endPosition = fmax((temp_tMax * NUMBER_OF_BINS + 2) * ((NUMBER_OF_BINS << 1) / NUMBER_OF_BINS) + 1,
                                   NUMBER_OF_BINS << 1);
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
        m_probabilityMask = std::make_shared<cv::Mat>(m_retardation->rows, m_retardation->cols, CV_32FC1);
        std::shared_ptr<cv::Mat> small_retardation = std::make_shared<cv::Mat>(m_retardation->rows/2, m_retardation->cols/2, CV_32FC1);
        std::shared_ptr<cv::Mat> small_transmittance = std::make_shared<cv::Mat>(m_transmittance->rows/2, m_transmittance->cols/2, CV_32FC1);
        MaskGeneration generation(small_retardation, small_transmittance);
        int numPixels = m_retardation->rows * m_retardation->cols;

        uint num_threads;
        #pragma omp parallel default(shared)
        num_threads = omp_get_num_threads();

        std::vector<std::mt19937> random_engines(num_threads);
        #pragma omp parallel for default(shared) schedule(static)
        for(unsigned i = 0; i < num_threads; ++i) {
            random_engines.at(i) = std::mt19937((clock() * i) % LONG_MAX);
        }
        std::uniform_int_distribution<int> distribution(0, numPixels);
        int selected_element;

        std::vector<float> above_tRet;
        std::vector<float> below_tRet;
        std::vector<float> above_tTra;
        std::vector<float> below_tTra;

        float t_ret, t_tra;

        for(unsigned i = 0; i < PROBABILITY_MASK_ITERATIONS; ++i) {
            std::cout << "\rProbability Mask Generation: Iteration " << i << " of " << PROBABILITY_MASK_ITERATIONS;
            std::flush(std::cout);
            // Fill transmittance and retardation with random pixels from our base images
            #pragma omp parallel for firstprivate(distribution, selected_element) schedule(static) default(shared)
            for(int y = 0; y < small_retardation->rows; ++y) {
                for (int x = 0; x < small_retardation->cols; ++x) {
                    selected_element = distribution(random_engines.at(omp_get_thread_num()));
                    small_retardation->at<float>(y, x) = m_retardation->at<float>(
                            selected_element / m_retardation->cols, selected_element % m_retardation->cols);
                    small_transmittance->at<float>(y, x) = m_transmittance->at<float>(
                            selected_element / m_transmittance->cols, selected_element % m_transmittance->cols);
                }
            }

            generation.setModalities(small_retardation, small_transmittance);
            generation.set_tMin(this->tMin());
            generation.set_tMax(this->tMax());

            t_ret = generation.tRet();
            if(t_ret >= this->tRet()) {
                above_tRet.push_back(t_ret);
            } else if(t_ret <= this->tRet()) {
                below_tRet.push_back(t_ret);
            }

            t_tra = generation.tTra();
            if(t_tra >= this->tTra()) {
                above_tTra.push_back(t_tra);
            } else if(t_tra <= this->tTra() && t_tra > 0) {
                below_tTra.push_back(t_tra);
            }
        }
        std::cout << std::endl;

        small_transmittance = nullptr;
        small_retardation = nullptr;
        generation.setModalities(nullptr, nullptr);

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

        float diffTra, diffRet;
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

